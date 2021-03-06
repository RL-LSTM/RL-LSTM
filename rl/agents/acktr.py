import copy
import os
import time
import operator
from functools import reduce

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from common.kfac import KFACOptimizer


class ACKTR(object):
	def __init__(self, actor_critic, rollout, learning_rate, 
				eps, num_processes, obs_shape, use_gae, gamma,
				 tau, recurrent_policy, num_mini_batch, cuda, 
				 log_interval, vis, env_name, log_dir, entropy_coef,
				 num_stack, num_steps, ppo_epoch, clip_param, 
				 max_grad_norm, alpha, save_dir, vis_interval, 
				 save_interval, num_updates, action_shape, value_loss_coef):
		'''
		set the argument values here
		'''

		self.eps = eps
		self.tau = tau
		self.cuda = cuda
		self.vis = vis
		self.gamma = gamma 
		self.alpha = alpha
		self.env_name = env_name
		self.num_updates = num_updates
		self.save_dir = save_dir
		self.use_gae = use_gae
		self.log_dir = log_dir
		self.action_shape = action_shape
		self.value_loss_coef = value_loss_coef
		self.num_stack = num_stack
		self.num_steps = num_steps
		self.ppo_epoch = ppo_epoch		
		self.obs_shape = obs_shape	
		self.clip_param = clip_param
		self.entropy_coef = entropy_coef
		self.log_interval = log_interval
		self.vis_interval = vis_interval	
		self.learning_rate = learning_rate
		self.num_processes = num_processes
		self.max_grad_norm = max_grad_norm		
		self.save_interval = save_interval
		self.num_mini_batch = num_mini_batch
		self.actor_critic = copy.deepcopy(actor_critic)
		self.recurrent_policy = recurrent_policy
		self.rollouts = copy.deepcopy(rollout)
		
		self.final_rewards = torch.zeros([self.num_processes, 1])
		self.episode_rewards = torch.zeros([self.num_processes, 1])
		self.current_obs = torch.zeros(self.num_processes, *self.obs_shape)
		self.optimizer = KFACOptimizer(self.actor_critic)

	def modelsize(self):
		modelSize = 0
		for p in self.actor_critic.parameters():
			pSize = reduce(operator.mul, p.size(), 1)
			modelSize += pSize
		return modelSize

	def update_current_obs(self, obs, envs):		
		shape_dim0 = envs.observation_space.shape[0]
		obs = torch.from_numpy(obs).float()
		if self.num_stack > 1:
			self.current_obs[:, :-shape_dim0] = self.current_obs[:, shape_dim0:]
		self.current_obs[:, -shape_dim0:] = obs

	def save_weights(self, envs):
		save_path = os.path.join(self.save_dir, 'acktr')
		try:
			os.makedirs(save_path)
		except OSError:
			pass
			
		if self.cuda: 
			save_model  = copy.deepcopy(self.actor_critic).cpu()
		else : 
			save_model = copy.deepcopy(self.actor_critic)
		save_model = [save_model,
				hasattr(envs, 'ob_rms') and envs.ob_rms or None]
		torch.save(save_model, os.path.join(save_path, self.env_name + ".pt"))

	def train(self, envs):
		if self.cuda:
			self.current_obs = self.current_obs.cuda()
			self.rollouts.cuda()
		start = time.time()
		for j in range(self.num_updates):
			for step in range(self.num_steps):
				# Sample actions
				value, action, action_log_prob, states = self.actor_critic.act(
					Variable(self.rollouts.observations[step], volatile=True),
					Variable(self.rollouts.states[step], volatile=True),
					Variable(self.rollouts.masks[step], volatile=True)
				)
				cpu_actions = action.data.squeeze(1).cpu().numpy()

				# Obser reward and next obs
				obs, reward, done, info = envs.step(cpu_actions)
				reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
				self.episode_rewards += reward

				# If done then clean the history of observations.
				masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
				self.final_rewards *= masks
				self.final_rewards += (1 - masks) * self.episode_rewards
				self.episode_rewards *= masks

				if self.cuda:
					masks = masks.cuda()

				if self.current_obs.dim() == 4:
					self.current_obs *= masks.unsqueeze(2).unsqueeze(2)
				else:
					self.current_obs *= masks

				self.update_current_obs(obs, envs)
				self.rollouts.insert(step, self.current_obs, states.data, action.data, action_log_prob.data, value.data, reward, masks)

			next_value = self.actor_critic(
				Variable(self.rollouts.observations[-1], volatile=True),
				Variable(self.rollouts.states[-1], volatile=True),
				Variable(self.rollouts.masks[-1], volatile=True)
			)[0].data

			self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.tau)

			values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
				Variable(self.rollouts.observations[:-1].view(-1, *self.obs_shape)),
				Variable(self.rollouts.states[:-1].view(-1, self.actor_critic.state_size)),
				Variable(self.rollouts.masks[:-1].view(-1, 1)),
				Variable(self.rollouts.actions.view(-1, self.action_shape))
			)

			values = values.view(self.num_steps, self.num_processes, 1)
			action_log_probs = action_log_probs.view(self.num_steps, self.num_processes, 1)

			advantages = Variable(self.rollouts.returns[:-1]) - values
			value_loss = advantages.pow(2).mean()

			action_loss = -(Variable(advantages.data) * action_log_probs).mean()

			self.actor_critic.zero_grad()
			pg_fisher_loss = -action_log_probs.mean()

			value_noise = Variable(torch.randn(values.size()))
			if self.cuda:
				value_noise = value_noise.cuda()

			sample_values = values + value_noise
			vf_fisher_loss = -(values - Variable(sample_values.data)).pow(2).mean()

			fisher_loss = pg_fisher_loss + vf_fisher_loss
			self.optimizer.acc_stats = True
			fisher_loss.backward(retain_graph=True)
			self.optimizer.acc_stats = False

			self.optimizer.zero_grad()
			(value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
			self.optimizer.step()

			self.rollouts.after_update()

			if j % self.save_interval == 0 and self.save_dir != "":
				self.save_weights(envs)

			if j % self.log_interval == 0:
				end = time.time()
				total_num_steps = (j + 1) * self.num_processes * self.num_steps
				print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
					format(j, total_num_steps,
						   int(total_num_steps / (end - start)),
						   self.final_rewards.mean(),
						   self.final_rewards.median(),
						   self.final_rewards.min(),
						   self.final_rewards.max(), dist_entropy.data[0],
						   value_loss.data[0], action_loss.data[0]))
			if self.vis and j % self.vis_interval == 0:
				try:
					# Sometimes monitor doesn't properly flush the outputs
					win = visdom_plot(viz, win, self.log_dir, self.env_name, 'acktr')
				except IOError:
					pass



			
