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

from rl.agents import DDQNAgent
from rl.models.ddqn import CriticNetwork
from rl.common.replaybuffer import ReplayBuffer

from GridWorldSimon import gameEnv

def main(args):
    CUDA = torch.cuda.is_available()
    OUTPUT_RESULTS_DIR = './saver'
    # ENVIRONMENT = 'CartPole-v1'
    ENVIRONMENT = 'GridWorldSimon-v1'
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "DQN", ENVIRONMENT, TIMESTAMP)

    # env = gym.make(ENVIRONMENT) #TODO
    # env = wrappers.Monitor(env, SUMMARY_DIR, force=True) #TODO
    env = gameEnv(size=5,startDelay=2)
    state_dim = 84*84*3#env.observation_space.shape[0] #TODO
    action_dim = env.action_space.n #TODO

    critic = CriticNetwork(state_dim, action_dim, learning_rate=args.learning_rate, epsilon=args.epsilon,
                           seed=args.seed, batch_size=args.batch_size, tau=args.tau)

    if CUDA:
        critic = critic.cuda()

    replay_buffer = ReplayBuffer(args.bufferlength)

    agent = DDQNAgent(critic, replay_buffer, episode_len=args.episode_len,
                      episode_steps=args.episode_steps, epsilon=args.epsilon, epsilon_decay=args.epsilon_decay,
                      batch_size=args.batch_size, gamma=args.gamma, seed=args.seed)

    if args.is_train:
        agent.train(env)
        agent.save_critic_weights(save_dir=OUTPUT_RESULTS_DIR, filename=args.actor_weights)
    else:
        agent.load_critic_weights(save_dir=OUTPUT_RESULTS_DIR, filename=args.actor_weights)
        agent.test(env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDPG working code for classical control tasks')
    parser.add_argument('--seed', type=int, default=1234, help='random seed to use. Default=1234')
    parser.add_argument('--tau', type=float, default=0.001, help='adaptability')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='critic learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='critic learning rate')
    parser.add_argument('--bufferlength', type=float, default=2000, help='buffer size in replay buffer')
    parser.add_argument('--l2_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--episode_len', type=int, default=1000, help='episodic lengths')
    parser.add_argument('--episode_steps', type=int, default=1000, help='steps per episode')
    parser.add_argument('--epsilon', type=float, default=0.01, help='noide standard deviation')
    parser.add_argument('--epsilon_decay', type=float, default=0.999, help='noide standard deviation')
    parser.add_argument('--is_train', type=bool, default=True, help='train mode or test mode. Default is test mode')
    parser.add_argument('--actor_weights', type=str, default='ddqn_cartpole',
                        help='Filename of actor weights. Default is actor_pendulum')
    args = parser.parse_args()

    main(args)
