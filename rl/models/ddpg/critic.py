import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init

class CriticNetwork(nn.Module):
	def __init__(self, state_dim, action_dim, action_bound, learning_rate, tau, L2_decay,seed):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_bound = action_bound
		self.learning_rate = learning_rate
		self.tau = tau
		self.seed = seed
		self.L2_decay = L2_decay
		super(CriticNetwork, self).__init__()

		self.layer1 = nn.Linear(self.state_dim,400); 
		n = weight_init._calculate_fan_in_and_fan_out(self.layer1.weight)[0]
		torch.manual_seed(self.seed)
		self.layer1.weight.data.normal_(0.0,math.sqrt(2./n))
		torch.manual_seed(self.seed)
		self.layer1.bias.data.normal_(0.0,math.sqrt(2./n))
		self.layer2 = nn.Linear(400,300)
		n = weight_init._calculate_fan_in_and_fan_out(self.layer2.weight)[0]
		torch.manual_seed(self.seed)
		self.layer2.weight.data.normal_(0.0,math.sqrt(2./n))
		torch.manual_seed(self.seed)
		self.layer2.bias.data.normal_(0.0,math.sqrt(2./n))
		self.layer3 = nn.Linear(action_dim,300,bias = False)
		n = weight_init._calculate_fan_in_and_fan_out(self.layer3.weight)[0]
		torch.manual_seed(self.seed)
		self.layer3.weight.data.normal_(0.0,math.sqrt(2./n))
		self.layer4 = nn.Linear(300,action_dim)
		torch.manual_seed(self.seed)
		self.layer4.weight.data.uniform_(-0.003,0.003)
		torch.manual_seed(self.seed)
		self.layer4.bias.data.uniform_(-0.003,0.003)
		
		self.loss_fn = torch.nn.MSELoss(size_average=True)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,weight_decay = L2_decay)

	def forward(self,state,action):
		y = F.relu(self.layer1(state))
		y = F.relu(self.layer2(y) + self.layer3(action))
		q_value = self.layer4(y)
		return state, action, q_value

	def predict_target(self,state,action):
		state,action,q_value = self.forward(state,action)
		return q_value

	def train(self,states,action,y):
		self.optimizer.zero_grad()
		q_value = self.predict_target(states,action)
		loss = self.loss_fn(q_value,y)
		loss_prev = 0.0
		loss.backward()
		self.optimizer.step()
		loss_prev = loss
		q_value = self.predict_target(states,action)
		loss = self.loss_fn(q_value,y)
		self.optimizer.zero_grad()

	def update_target_network(self,critic):
		for weight,target_weight in zip(self.parameters(),critic.parameters()):
			weight.data = (1-self.tau)*weight.data +  (self.tau)*target_weight.data
