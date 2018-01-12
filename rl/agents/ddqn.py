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

class DDQNAgent: 
	def __init__(self, critic, replay_buffer, episode_len = 1000, episode_steps=1000, epsilon = 0.01, epsilon_decay = 0.999, batch_size = 64, gamma = 0.99, seed = 1234):
		self.critic = copy.deepcopy(critic)
		self.target_critic = copy.deepcopy(critic)
		self.replay_buffer = copy.deepcopy(replay_buffer)
		self.episode_len = episode_len
		self.episode_steps = episode_steps
		self.epsilon = epsilon
		self.batch_size = batch_size
		self.epsilon_decay = epsilon_decay
		self.gamma = gamma
		self.seed = seed

	def save_critic_weights(self, save_dir='../saver', filename='sample_critic'):
		PATH = save_dir + '/' + filename
		torch.save(self.critic.state_dict(), PATH)		

	def load_critic_weights(self, save_dir='../saver', filename='sample_critic'):
		PATH = save_dir + '/' + filename
		self.critic.load_state_dict(torch.load(PATH))

	def take_action(self, action, nb_actions, epsilon_t):
		output_action = np.ones(nb_actions)
		output_prob = output_action*epsilon_t*(1./nb_actions)
		max_argument = np.argmax(action)
		output_prob[max_argument] += 1 - epsilon_t
		return np.random.choice(range(nb_actions), p=output_prob)


	def train(self, env):
		CUDA = torch.cuda.is_available()
		epsilon_t = 1.0
		for i in tqdm(range(self.episode_len)):
			state = time()
			env.seed(self.seed + i)
			s = env.reset()
			terminal = False
			while not terminal:
				env.render()
				input_state  = np.reshape(s, (1, self.critic.state_dim))
				input_state = torch.from_numpy(input_state)
				dtype = torch.FloatTensor
				input_state = Variable(input_state.type(dtype),requires_grad=True)
				if CUDA:
					input_state = input_state.cuda()
				a = self.critic(input_state) 
				a = a.data.cpu().numpy()			
				a = self.take_action(a[0], env.action_space.n, epsilon_t)
				s2, r, terminal, info = env.step(a)
				self.replay_buffer.add(np.reshape(s, (self.critic.state_dim,)),
								  a,
								  r, terminal, np.reshape(s2, (self.critic.state_dim,)))
				if epsilon_t > self.epsilon:
					epsilon_t = epsilon_t*self.epsilon_decay
				else : 
					epsilon_t = self.epsilon
				if self.replay_buffer.size() > self.batch_size:	
					s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay_buffer.sample_batch(self.batch_size)
					s2_batch = torch.from_numpy(s2_batch)
					s2_batch = Variable(s2_batch.type(torch.FloatTensor),requires_grad=False)

					if CUDA:
						s2_batch = s2_batch.cuda()
					critic_output = self.target_critic(s2_batch)
					critic_output_model = self.critic(s2_batch)
					arg_out = torch.max(critic_output_model,1)[1]
					arg_out = arg_out.data.numpy().astype(int)
					critic_output =  critic_output[:,arg_out]
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
					self.target_critic = copy.deepcopy(self.critic)
					self.critic.train(s_batch, a_batch, y)	

				else:
					loss = 0

				s = s2
				if terminal:
					break	

	def test(self, env):
		CUDA = torch.cuda.is_available()
		for i in tqdm(range(self.episode_len)):
			state = time()
			s = env.reset()
			terminal = False
			while not terminal:
				env.render()
				input_state  = np.reshape(s, (1, self.critic.state_dim))
				input_state = torch.from_numpy(input_state)
				dtype = torch.FloatTensor
				input_state = Variable(input_state.type(dtype),requires_grad=True)

				if CUDA:
					input_state = input_state.cuda()

				a = self.critic(input_state) 
				a = a.data.cpu().numpy()	
				a = np.argmax(a)	
				s, r, terminal, info = env.step(a)	
				if terminal:
					break	










