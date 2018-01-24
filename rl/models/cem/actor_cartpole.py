import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init


class ActorNetwork(nn.Module):
	def __init__(self, state_dim, action_dim):
		self.state_dim = state_dim
		self.action_dim = action_dim
		super(ActorNetwork, self).__init__()
	
		self.layer1 = nn.Linear(self.state_dim, 24)
		self.layer2 = nn.Linear(24,24)
		self.layer3 = nn.Linear(24,self.action_dim)
		self.sizes = [torch.numel(w.data) for w in self.parameters()]
		self.shape = [np.shape(w.data.numpy()) for w in self.parameters()]
		self.num_weights = sum(self.sizes)

	def init_weight(self, mean = 0.0, std = 1.0):
		pos = 0
		for weight, shape in zip(self.parameters(), self.shape):			
			if (np.isscalar(mean) and np.isscalar(std)):

				mean_t = torch.from_numpy(mean*np.ones(shape))
				mean_t = mean_t.type(torch.FloatTensor)
				std_t = torch.from_numpy(std*np.ones(shape))
				std_t = std_t.type(torch.FloatTensor)
				weight.data =  torch.normal(mean_t, std_t)
			else : 
				mean_t = np.array(mean[pos:pos+np.prod(shape)])
				mean_t = torch.from_numpy(mean_t.reshape(shape))
				mean_t = mean_t.type(torch.FloatTensor)
				std_t = np.array(std[pos:pos+np.prod(shape)])
				std_t = torch.from_numpy(std_t.reshape(shape))
				std_t = std_t.type(torch.FloatTensor)				
				weight.data =  torch.normal(mean_t, std_t)
				pos +=  np.prod(shape)




	def forward(self, x):

		y = F.relu(self.layer1(x))
		y = F.relu(self.layer2(y))
		y = self.layer3(y)
		y = y - torch.max(y)
		y = torch.exp(y)
		action = torch.multinomial(y/torch.sum(y.data))
		return action

	def get_weights_flat(self):

		weights_flat = np.zeros(self.num_weights)
		pos = 0
		for weights, shape in zip(self.parameters(), self.sizes):
			weights_flat[pos:pos+shape] = weights.data.numpy().reshape(-1)
			pos = pos + shape
		return weights_flat

	def get_weights_list(self,weights_flat):

		weights = []
		pos = 0
		for size in enumerate(self.shape):
			arr = weights_flat[pos:pos+size].reshape(size)
			weights.append(arr)
			pos += size
		return weights   




