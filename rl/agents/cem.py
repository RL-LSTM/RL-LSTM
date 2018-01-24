# coding: utf-8
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

class CEMAgent:
	def __init__(self, actor, action_dim, batch_size=64, ep_len = 100,
				 elite_frac=0.05, noise_decay_const=0.0, noise_ampl=0.0):

				 self.actor = copy.deepcopy(actor)
				 self.ep_len = ep_len
				 self.action_dim = action_dim
				 self.batch_size = batch_size
				 self.elite_frac = elite_frac	
				 self.noise_decay_const = noise_decay_const
				 self.noise_ampl = noise_ampl

				 self.init_mean = 0.0
				 self.init_std  = 1.0

	def save_actor_weights(self, save_dir='../saver', filename='sample_actor'):
		PATH = save_dir + '/' + filename
		torch.save(self.actor.state_dict(), PATH)

	def load_actor_weights(self, save_dir='../saver', filename='sample_actor'):
		PATH = save_dir + '/' + filename
		self.actor.load_state_dict(torch.load(PATH))
	
	def train(self, env):
		CUDA = torch.cuda.is_available()
		params = []
		rewards = []
		mean_t = 0.0
		std_t = 1.0
		num_best = int(self.batch_size*self.elite_frac)
		for i in tqdm(range(self.ep_len)):

			mean_rewards = 0.0

			for j in range(self.batch_size):

				self.actor.init_weight(mean = mean_t, std = std_t)
				s = env.reset()
				done = False
				rew = 0

				while not done:
					env.render()
					input_state  = np.reshape(s, (1, self.actor.state_dim))
					input_state = torch.from_numpy(input_state)
					dtype = torch.FloatTensor
					input_state = Variable(input_state.type(dtype),requires_grad=False)
					if CUDA:
						input_state = input_state.cuda()
					a = self.actor(input_state) 
					a = a.data.cpu().numpy()
					a =  a[0][0]
					s, r, done, info = env.step(a)
					rew += r
				mean_rewards += rew
				rewards.append(rew)
				flat_weights = self.actor.get_weights_flat()
				params.append(flat_weights)
			tqdm.write("iterations: %.4f, mean reward: %.4f" %(i, mean_rewards/self.batch_size))
			best_idx = np.argsort(np.array(rewards))[-num_best:]
			best = np.vstack([params[i] for i in best_idx])
			min_std = self.noise_ampl * np.exp(-i * self.noise_decay_const)
			mean_t = np.mean(best, axis=0)
			std_t = np.std(best, axis=0) + min_std
	def test(self, env):
		CUDA = torch.cuda.is_available()
		mean_rewards = 0.0
		for i in tqdm(range(self.ep_len)):

				s = env.reset()
				done = False
				rew = 0
				while not done:
					env.render()
					input_state  = np.reshape(s, (1, self.actor.state_dim))
					input_state = torch.from_numpy(input_state)
					dtype = torch.FloatTensor
					input_state = Variable(input_state.type(dtype),requires_grad=False)
					if CUDA:
						input_state = input_state.cuda()
					a = self.actor(input_state) 
					a = a.data.cpu().numpy()
					a =  a[0][0]
					s, r, done, info = env.step(a)
					rew += r
				mean_rewards += rew	
		tqdm.write("iterations: %.4f, mean reward: %.4f" %(i, mean_rewards/self.ep_len))











		




