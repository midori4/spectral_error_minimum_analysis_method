import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
	def __init__(self, in_dim, out_dim, numlayer, numunit):
		super(MLP, self).__init__()
		in_sizes = [in_dim] + [numunit] * (numlayer - 1)
		out_sizes = [numunit] * numlayer
		self.layers = nn.ModuleList(
			[nn.Linear(in_size, out_size) for (in_size, out_size)
			in zip(in_sizes, out_sizes)])
		self.last_linear = nn.Linear(numunit, out_dim)
	
	def forward(self, x):
		for layer in self.layers:
			x = F.relu(layer(x))
		x = self.last_linear(x)
		return x

class CNN(nn.Module):
	def __init__(self, filtersize, stride, padsize, numchannel):
		super(CNN, self).__init__()
		self.layers = nn.ModuleList([ \
			nn.Conv1d(in_channels=1, out_channels=numchannel, kernel_size=filtersize, stride=stride, padding=padsize), \
			nn.Conv1d(in_channels=numchannel, out_channels=numchannel, kernel_size=filtersize, stride=stride, padding=padsize), \
			nn.Conv1d(in_channels=numchannel, out_channels=numchannel, kernel_size=filtersize, stride=stride, padding=padsize), \
			nn.Conv1d(in_channels=numchannel, out_channels=numchannel, kernel_size=filtersize, stride=stride, padding=padsize), \
			nn.Conv1d(in_channels=numchannel, out_channels=numchannel, kernel_size=filtersize, stride=stride, padding=padsize), \
			nn.Conv1d(in_channels=numchannel, out_channels=1, kernel_size=filtersize, stride=stride, padding=padsize) \
		])
	
	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x
