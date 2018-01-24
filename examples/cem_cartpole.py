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

from rl.agents import CEMAgent
from rl.models.cem import ActorNetwork

def main(args):
	CUDA = torch.cuda.is_available()
	OUTPUT_RESULTS_DIR = './saver'
	ENVIRONMENT = 'CartPole-v0'
	TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
	SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "CEM", ENVIRONMENT, TIMESTAMP)
	
	env = gym.make(ENVIRONMENT)
	env = wrappers.Monitor(env, SUMMARY_DIR, force=True)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n

	actor = ActorNetwork(state_dim, action_dim)

	if CUDA: 
		actor = actor.cuda()


	agent = CEMAgent(actor, action_dim, batch_size=args.batch_size, ep_len = args.episode_len,
				 elite_frac=args.elite_frac, noise_decay_const=args.noise_decay_const, noise_ampl=args.noise_ampl)

	if args.is_train:
		agent.train(env)
		agent.save_actor_weights(save_dir=OUTPUT_RESULTS_DIR, filename=args.actor_weights)	
	else:
		agent.load_actor_weights(save_dir=OUTPUT_RESULTS_DIR, filename=args.actor_weights)
		agent.test(env)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='CEM working code for classical control tasks')
	parser.add_argument('--batch_size', type=int, default=64, help='critic learning rate')
	parser.add_argument('--episode_len', type=float, default=50, help='buffer size in replay buffer')
	parser.add_argument('--elite_frac', type=float, default=0.05, help='noide standard deviation')
	parser.add_argument('--noise_decay_const', type=float, default=0.00, help='noise standard deviation')
	parser.add_argument('--noise_ampl', type=float, default=0.00, help='noise standard deviation')
	parser.add_argument('--is_train', type=bool, default=False, help='train mode or test mode. Default is test mode')
	parser.add_argument('--actor_weights', type=str, default='cem_cartpole', help='Filename of actor weights. Default is actor_cartpole')
	args = parser.parse_args()

	main(args)
