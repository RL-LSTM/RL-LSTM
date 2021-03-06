import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init

class CriticNetwork(nn.Module):
	def __init__(self, state_dim, action_dim, learning_rate, epsilon, seed, batch_size, tau, nhid = 300):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.learning_rate = learning_rate 
		self.epsilon = epsilon
		self.seed = seed
		self.batch_size = batch_size
		self.tau = tau
		self.duel_enable = False
		self.duel_type = False
		self.nhid = nhid

		super(CriticNetwork, self).__init__()
	
		self.layer1 = nn.Linear(self.state_dim,24)
		n = weight_init._calculate_fan_in_and_fan_out(self.layer1.weight)[0]
		torch.manual_seed(self.seed)		
		self.layer1.weight.data.uniform_(-math.sqrt(6./n), math.sqrt(6./n))		
		self.layer2 = nn.Linear(24,24)
		n = weight_init._calculate_fan_in_and_fan_out(self.layer2.weight)[0]
		torch.manual_seed(self.seed)		
		self.layer2.weight.data.uniform_(-math.sqrt(6./n), math.sqrt(6./n))

		# RL-LSTM
		self.layerLSTM = torch.nn.LSTMCell(24,nhid,bias=True)
		self.hiddenLSTM = self.init_hidden(self.batch_size)
		self.init_lstmCellWeights()
		self.layerLinearPostLSTM = nn.Linear(nhid,24)
		n = weight_init._calculate_fan_in_and_fan_out(self.layerLinearPostLSTM.weight)[0]
		torch.manual_seed(self.seed)
		self.layerLinearPostLSTM.weight.data.uniform_(-math.sqrt(6. / n), math.sqrt(6. / n))


		self.layer3 = nn.Linear(24,action_dim)
		n = weight_init._calculate_fan_in_and_fan_out(self.layer3.weight)[0]
		torch.manual_seed(self.seed)		
		self.layer3.weight.data.uniform_(-math.sqrt(6./n), math.sqrt(6./n))		

		self.loss_fn = torch.nn.MSELoss(size_average=True)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay = 0.01)		

	def forward(self, x):
		y = F.relu(self.layer1(x))
		y = F.relu(self.layer2(y))
		self.hiddenLSTM = self.layerLSTM(y,self.hiddenLSTM)
		y = F.relu(self.layerLinearPostLSTM(self.hiddenLSTM[0]))
		y = self.layer3(y)
		return y


	def train(self, states, actions, y):
		self.optimizer.zero_grad()
		q_value = self.forward(states)
		actions = actions.data.cpu().numpy().astype(int)
		range_array = np.array(range(self.batch_size))
		q_value = q_value[:, actions]
		loss = self.loss_fn(q_value,y)
		loss.backward()
		self.optimizer.step()
		# TODO WTF loss2 important to?
		q_value = self.forward(states)
		q_value = q_value[:, actions]
		loss2 = self.loss_fn(q_value,y)						
		self.optimizer.zero_grad()

	def update_target_weights(self, critic):

		for weight,target_weight in zip(self.parameters(),critic.parameters()):
			weight.data = (1-self.tau)*weight.data +  (self.tau)*target_weight.data

	def init_hidden(self, bsz):
		weight = next(self.parameters())
		return (weight.new_zeros(self.nlayers, bsz, self.nhid),
					weight.new_zeros(self.nlayers, bsz, self.nhid))

	def init_lstmCellWeights(self):
		# the xavier initialization here is different than the one at the original code of critic_cartpole.
		# taken from: https://discuss.pytorch.org/t/initializing-parameters-of-a-multi-layer-lstm/5791
		for name, param in self.layerLSTM.named_parameters():
			if 'bias' in name:
				nn.init.constant(param, 1.0)
			elif 'weight' in name:
				nn.init.xavier_normal(param)


