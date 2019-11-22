"""Train mapping for floor-function.
usage: train.py [-h|--help] (-o <outdir>)

options:
  -h --help                     Show this message and exit.
  -o <outdir>                   Output directory for saving estimated result (required).
"""
from docopt import docopt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from tqdm import tqdm
from os.path import join

import matplotlib.pyplot as plt

class MLP(nn.Module):
	def __init__(self):
		super(MLP, self).__init__()
		self.layers = nn.ModuleList(
			[nn.Linear(1, 1024), nn.Linear(1024, 1024)]
		)
		self.last_layer = nn.Linear(1024, 1)
	
	def forward(self, x):
		for layer in self.layers:
			x = F.relu(layer(x))
		return self.last_layer(x)

# Training DNN...
def batch_train(model, optimizer, criterion, x, y, nepoch, outdir):
	model.train()
	trainloss_log = ""
	
	# training
	for epoch in tqdm(range(1, nepoch+1)):
		pred_y = model(Variable(x))
		loss = criterion(pred_y, Variable(y))
		
		# Reset optimizer state
		optimizer.zero_grad()
				
		# backpropagation
		loss.backward()
		optimizer.step()
		
		# standard output loss
		tmploss = str(epoch) + "epoch" + "\t" + str(loss.item()) + "\n"
		print(tmploss)
		
		trainloss_log += str(loss.item()) + "\n"
		if epoch % 1000 == 0:
			model.eval()
			pred_y = model(x)
			plt.plot(x.cpu().data.numpy(), y.cpu().data.numpy(), 'k', linewidth=0.5, label="Target")
			plt.plot(x.cpu().data.numpy(), pred_y.cpu().data.numpy(), 'r', linewidth=0.5, label="Output of MLP")
			plt.xlim(0, 10)
			plt.xlabel('Input')
			plt.legend()
			plt.savefig(join(outdir,str(epoch).zfill(6)+".eps"))
			plt.close()
			model.train()
	return trainloss_log


if __name__ == '__main__':
	args = docopt(__doc__)
	print("Command line args\n", args)
	outdir = args["-o"]
	
	gpuid = 0
	dtype = torch.float
	device = torch.device("cuda:"+str(gpuid) if gpuid>=0 else "cpu")
	
	x = np.arange(0, 1000) * 0.01
	y = np.floor(x)
	x = torch.from_numpy(x[:,np.newaxis]).to(dtype).to(device)
	y = torch.from_numpy(y[:,np.newaxis]).to(dtype).to(device)
	
	
	model = MLP()
	model = model.to(device)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters())
	
	losslog = batch_train(model, optimizer, criterion, x, y, 200000, outdir)
	with open(join(outdir, "loss.txt"), mode="w") as f:
		f.write(losslog)
	
	sys.exit(0)
