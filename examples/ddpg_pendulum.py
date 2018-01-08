import os
import sys
import gym
import math
import copy
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from time import time
from gym import wrappers
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as weight_init

from rl.agents import DDPGAgent
from rl.common.ounoise import OUNoise
from rl.models.ddpg import ActorNetwork
from rl.models.ddpg import CriticNetwork
from rl.common.replaybuffer import ReplayBuffer

def main(args):
	CUDA = torch.cuda.is_available()
	OUTPUT_RESULTS_DIR = './saver'
	ENVIRONMENT = 'SemisuperPendulumRandom-v0'
	TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
	SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "DDPG", ENVIRONMENT, TIMESTAMP)
	
	env = gym.make(ENVIRONMENT)
	env = wrappers.Monitor(env, SUMMARY_DIR, force=True)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	action_bound = env.action_space.high

	actor = ActorNetwork(state_dim, action_dim, action_bound, args.actor_lr, args.tau, args.seed)
	target_actor = ActorNetwork(state_dim, action_dim, action_bound, args.actor_lr, args.tau, args.seed)
	critic = CriticNetwork(state_dim, action_dim, action_bound, args.critic_lr, args.tau, args.l2_decay, args.seed)
	target_critic = CriticNetwork(state_dim, action_dim, action_bound, args.critic_lr, args.tau, args.l2_decay, args.seed)

	if CUDA: 
		actor = actor.cuda()
		target_actor = target_actor.cuda()
		critic = critic.cuda()
		target_critic = target_critic.cuda()

	replay_buffer = ReplayBuffer(args.bufferlength, args.seed)

	agent = DDPGAgent(actor, target_actor, critic, target_critic, replay_buffer, 
					  batch_size=args.batch_size, gamma = args.gamma, seed=args.seed, 
					  episode_len=args.episode_len, episode_steps=args.episode_steps,
					  noise_mean = args.noise_mean, noise_th=args.noise_th, noise_std=args.noise_std,
					  noise_decay=args.noise_decay)

	if args.is_train:
		agent.train(env)
		agent.save_actor_weights(save_dir=OUTPUT_RESULTS_DIR, filename=args.actor_weights)
	else:
		agent.load_actor_weights(save_dir=OUTPUT_RESULTS_DIR, filename=args.actor_weights)
		agent.test(env)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='DDPG working code for classical control tasks')
	parser.add_argument('--seed', type=int, default=1234, help='random seed to use. Default=1234')
	parser.add_argument('--actor_lr', type=float, default=0.0001, help='actor learning rate')
	parser.add_argument('--critic_lr', type=float, default=0.001, help='critic learning rate')
	parser.add_argument('--batch_size', type=int, default=64, help='critic learning rate')
	parser.add_argument('--bufferlength', type=float, default=1000000, help='buffer size in replay buffer')
	parser.add_argument('--l2_decay', type=float, default=0.01, help='weight decay')
	parser.add_argument('--tau', type=float, default=0.001, help='adaptability')
	parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
	parser.add_argument('--episode_len', type=int, default=10000, help='episodic lengths')
	parser.add_argument('--episode_steps', type=int, default=1000, help='steps per episode')
	parser.add_argument('--noise_mean', type=float, default=0.0, help='noise mean')
	parser.add_argument('--noise_th', type=float, default=0.15, help='noise theta')
	parser.add_argument('--noise_std', type=float, default=0.20, help='noide standard deviation')
	parser.add_argument('--noise_decay', type=int, default=25, help='linear decrease in noise')
	parser.add_argument('--is_train', type=bool, default=False, help='train mode or test mode. Default is test mode')
	parser.add_argument('--actor_weights', type=str, default='actor_pendulum', help='Filename of actor weights. Default is actor_pendulum')
	args = parser.parse_args()

	main(args)
