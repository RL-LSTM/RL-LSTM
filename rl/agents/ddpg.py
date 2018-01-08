import os
import gym
import sys
import math
import copy
import random
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from gym import wrappers
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable
from rl.common.ounoise import OUNoise

class DDPGAgent:
	def __init__(self, actor, target_actor, critic,target_critic, replay_buffer,
				 batch_size=64, gamma = 0.99, seed=1234, episode_len=1000, episode_steps=1000,
				 noise_mean = 0.0, noise_th=0.15, noise_std=0.20, noise_decay=25):
		self.actor = copy.deepcopy(actor)
		self.target_actor = copy.deepcopy(target_actor)
		self.critic = copy.deepcopy(critic)
		self.target_critic = copy.deepcopy(target_critic)
		self.replay_buffer = copy.deepcopy(replay_buffer)
		self.batch_size = batch_size
		self.gamma = gamma
		self.seed = seed
		self.episode_len = episode_len
		self.episode_steps = episode_steps
		self.noise_mean = noise_mean
		self.noise_th = noise_th
		self.noise_std = noise_std
		self.noise_decay = noise_decay

	def save_actor_weights(self, save_dir='../saver', filename='sample_actor'):
		PATH = save_dir + '/' + filename
		torch.save(self.actor.state_dict(), PATH)

	def load_actor_weights(self, save_dir='../saver', filename='sample_actor'):
		PATH = save_dir + '/' + filename
		self.actor.load_state_dict(torch.load(PATH))

	def save_critic_weights(self, save_dir='../saver', filename='sample_critic'):
		PATH = save_dir + '/' + filename
		torch.save(self.critic.state_dict(), PATH)		

	def load_critic_weights(self, save_dir='../saver', filename='sample_critic'):
		PATH = save_dir + '/' + filename
		self.critic.load_state_dict(torch.load(PATH))

	def train(self,env):
		CUDA = torch.cuda.is_available()
		for i in tqdm(range(self.episode_len)):
			start = time()
			ep_rewards = []
			ep_action_dist = []
			ep_loss = []
			env.seed(self.seed + i)
			s = env.reset()
			exploration_noise = OUNoise(self.actor.action_dim, self.noise_mean, self.noise_th, self.noise_std, self.seed + i)

			for j in range(self.episode_steps):
				env.render()
				input_state  = np.reshape(s, (1, self.actor.state_dim))
				input_state = torch.from_numpy(input_state)
				dtype = torch.FloatTensor
				input_state = Variable(input_state.type(dtype),requires_grad=True)
				if CUDA:
					input_state = input_state.cuda()
				a = self.actor.predict(input_state) 
				epsilon = Variable(torch.FloatTensor(1).fill_(np.exp(-i/self.noise_decay)),requires_grad=False)
				if CUDA:
					epsilon = epsilon.cuda()
				noise_exp = epsilon * exploration_noise.noise()[0] / env.action_space.high[0]
				a += noise_exp.unsqueeze(0)
				a = torch.clamp(a, env.action_space.low[0], env.action_space.high[0])
				a = a.data.cpu().numpy()
				s2, r, terminal, info = env.step(a[0])
				ep_action_dist.append(a[0])
				self.replay_buffer.add(np.reshape(s, (self.actor.state_dim,)),
								  np.reshape(a, (self.actor.action_dim,)),
								  r, terminal, np.reshape(s2, (self.actor.state_dim,)))

				if self.replay_buffer.size() > self.batch_size:
					s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay_buffer.sample_batch(self.batch_size)
					s2_batch = torch.from_numpy(s2_batch)
					s2_batch = Variable(s2_batch.type(torch.FloatTensor),requires_grad=False)

					if CUDA:
						s2_batch = s2_batch.cuda()
					critic_output = self.target_critic.predict_target(s2_batch, self.target_actor.predict(s2_batch))
					critic_output = critic_output.data.cpu().numpy()
					y = r_batch + self.gamma * critic_output * ~t_batch
					s_batch = torch.from_numpy(s_batch)
					s_batch = Variable(s_batch.type(torch.FloatTensor),requires_grad=True)
					a_batch = torch.from_numpy(a_batch)
					a_batch = Variable(a_batch.type(torch.FloatTensor),requires_grad=True)
					y = torch.from_numpy(y)
					y = Variable(y.type(torch.FloatTensor),requires_grad=False)
					if CUDA: 
						s_batch = s_batch.cuda()
						a_batch = a_batch.cuda()
						y = y.cuda()
					self.critic.train(s_batch, a_batch, y)
					self.actor.train(s_batch, self.critic)
					self.target_actor.update_target_network(self.actor)
					self.target_critic.update_target_network(self.critic)
				else:
					loss = 0

				s = s2
				ep_rewards.append(r)
				ep_loss.append(loss)

				if terminal or j == self.episode_steps - 1:
					exploration_noise.reset()
					break

	def test(self,env):
		CUDA = torch.cuda.is_available()
		env.seed(1234)
		for i in tqdm(range(self.episode_len)):
			s = env.reset()
			for j in range(self.episode_steps):
				env.render()
				input_state  = np.reshape(s, (1, self.actor.state_dim))
				input_state = torch.from_numpy(input_state)
				dtype = torch.FloatTensor
				input_state = Variable(input_state.type(dtype),requires_grad=True)
				if CUDA:
					input_state = input_state.cuda()
				a = self.actor.predict(input_state) 
				a = a.data.cpu().numpy()
				s, r, terminal, info = env.step(a[0])    		
