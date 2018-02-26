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



class PPO(object):
	def __init__(self, actor_critic, rollout, learning_rate, 
				eps, num_processes, obs_shape, use_gae, gamma,
				 tau, recurrent_policy, num_mini_batch, cuda, 
				 log_interval, vis, env_name, log_dir, entropy_coef,
				 num_stack, num_steps, ppo_epoch, clip_param, 
				 max_grad_norm, save_dir, vis_interval, save_interval, 
				 num_updates, action_shape, value_loss_coef):
		'''
		set the argument values here
		'''

		self.eps = eps
		self.tau = tau
		self.cuda = cuda
		self.vis = vis
		self.gamma = gamma 
		self.num_updates = num_updates
		self.env_name = env_name
		self.save_dir = save_dir
		self.use_gae = use_gae
		self.log_dir = log_dir
		self.num_stack = num_stack
		self.num_steps = num_steps
		self.action_shape = action_shape
		self.value_loss_coef = value_loss_coef
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
		self.optimizer = optim.Adam(self.actor_critic.parameters(), self.learning_rate, eps=self.eps)

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
		save_path = os.path.join(self.save_dir, 'ppo')
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

			advantages = self.rollouts.returns[:-1] - self.rollouts.value_preds[:-1]
			advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

			for e in range(self.ppo_epoch):
				if self.recurrent_policy:
					data_generator = self.rollouts.recurrent_generator(advantages, self.num_mini_batch)
				else:
					data_generator = self.rollouts.feed_forward_generator(advantages, self.num_mini_batch)

				for sample in data_generator:
					observations_batch, states_batch, actions_batch, \
					   return_batch, masks_batch, old_action_log_probs_batch, \
							adv_targ = sample

					# Reshape to do in a single forward pass for all steps
					values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
						Variable(observations_batch),
						Variable(states_batch),
						Variable(masks_batch),
						Variable(actions_batch)
					)

					adv_targ = Variable(adv_targ)
					ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
					surr1 = ratio * adv_targ
					surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
					action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

					value_loss = (Variable(return_batch) - values).pow(2).mean()

					self.optimizer.zero_grad()
					(value_loss + action_loss - dist_entropy * self.entropy_coef).backward()
					nn.utils.clip_grad_norm(self.actor_critic.parameters(), self.max_grad_norm)
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
					win = visdom_plot(viz, win, self.log_dir, self.env_name, 'ppo')
				except IOError:
					pass



			
