import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init

class CriticNetwork(nn.Module):
	def __init__(self, state_dim, action_dim, learning_rate, epsilon, seed, batch_size, tau, duel_enable = False, duel_type = 'avg'):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.learning_rate = learning_rate 
		self.epsilon = epsilon
		self.seed = seed
		self.batch_size = batch_size
		self.tau = tau
		self.duel_enable = duel_enable
		self.duel_type = duel_type
		super(CriticNetwork, self).__init__()
	
		self.layer1 = nn.Linear(self.state_dim,24)
		n = weight_init._calculate_fan_in_and_fan_out(self.layer1.weight)[0]
		torch.manual_seed(self.seed)		
		self.layer1.weight.data.uniform_(-math.sqrt(6./n), math.sqrt(6./n))		
		self.layer2 = nn.Linear(24,24)
		n = weight_init._calculate_fan_in_and_fan_out(self.layer2.weight)[0]
		torch.manual_seed(self.seed)		
		self.layer2.weight.data.uniform_(-math.sqrt(6./n), math.sqrt(6./n))				
		if self.duel_enable:
			self.layer3 = nn.Linear(24,action_dim+1)
		else : 
			self.layer3 = nn.Linear(24,action_dim)
		n = weight_init._calculate_fan_in_and_fan_out(self.layer3.weight)[0]
		torch.manual_seed(self.seed)		
		self.layer3.weight.data.uniform_(-math.sqrt(6./n), math.sqrt(6./n))		

		self.loss_fn = torch.nn.MSELoss(size_average=True)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay = 0.01)		

	def forward(self, x):
		y = F.relu(self.layer1(x))
		y = F.relu(self.layer2(y))
		y = self.layer3(y)
		if self.duel_enable:		
			if self.duel_type == 'avg':
				output = y[:,0].unsqueeze(1) + (y[:, 1:] - torch.mean(y[:,1:],1, True))
				return output[:]

			elif self.duel_type == 'max':
				return y[:,0].unsqueeze(1) + (y[:, 1:] - torch.max(y[:,1:],1)[0].unsqueeze(1))

			elif self.duel_type == 'naive':
				return y[:,0].unsqueeze(1) + y[:, 1:]

			else :
				assert False, "dueling_type must be one of {'avg','max','naive'}"
		
		else : 
			return y


	def train(self, states, actions, y):
		self.optimizer.zero_grad()
		q_value = self.forward(states)
		actions = actions.data.numpy().astype(int)
		range_array = np.array(range(self.batch_size))
		index_range = np.arange(self.batch_size)
		index_range = np.reshape(index_range,(1, self.batch_size))
		q_value = q_value[index_range, actions]
		loss = self.loss_fn(q_value,y)
		loss.backward()
		self.optimizer.step()
		self.optimizer.zero_grad()

	def update_target_weights(self, critic):

		for weight,target_weight in zip(self.parameters(),critic.parameters()):
			weight.data = (1-self.tau)*weight.data +  (self.tau)*target_weight.data






