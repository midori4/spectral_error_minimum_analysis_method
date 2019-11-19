"""Train MLP which predict amplitude spectrum of pulse train from F0.

usage: train.py [-h|--help] (-f <filtersize>) (-s <stride>) (-p <padsize>) (-c <numchannel>) (-e <nepoch>) [-b <batchsize>] (--log <outlosslog>) (-m <outmodel>) (-g <gpuid>)

options:
  -h --help                 Show this message and exit.
  -f <filtersize>           The filter size for CNN (required).
  -s <stride>               Stride for CNN (required).
  -p <padsize>              zero padding size for CNN (required).
  -c <numchannel>           The number of channels (required).
  -e <nepoch>               The number of epochs (required).
  -b <batchsize>            Batch size. If you don't give this args batch learning will be done.
  --log <outlosslog>        The output train loss log file (required).
  -m <outmodel>             The output trained model parameters (required).
  -g <gpuid>                GPU id, if it is less than 0 this processing executed on CPU.
"""
from docopt import docopt

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from model import CNN

from tqdm import tqdm
import sys


class F02SP():
	def __init__(self, fftsize, fs):
		self.fftsize = fftsize
		self.N = self.fftsize // 2 + 1
		self.fs = fs
		self.nyquist_freq = fs // 2
		self.discrete_freq = self.nyquist_freq * np.arange(self.N) / (self.N - 1)
	
	def get_sp(self, f0):
		nlist = (self.nyquist_freq / f0).astype(np.int16) + 1
		sp = np.zeros((len(f0),self.N), dtype=np.float32)
		for frame, n in enumerate(nlist):
			for i in range(n):
				idx = np.abs(self.discrete_freq - i * f0[frame]).argmin()
				sp[frame][idx] = 1
				# sp[frame][idx] = self.fftsize / np.sqrt(1 + self.fs / f0[frame])
		return sp, self.discrete_freq

# Training DNN...
def batch_train(model, optimizer, criterion, x, y, nepoch):
	model.train()
	trainloss_log = ""
	
	# training
	for epoch in tqdm(range(1, nepoch+1)):
		pred_y = model(Variable(x))
		loss = criterion(pred_y[:,:,1:], Variable(y)[:,:,1:])
		
		# Reser optimizer state
		optimizer.zero_grad()
				
		# backpropagation
		loss.backward()
		optimizer.step()
		
		# standard output loss
		tmploss = str(epoch) + "epoch" + "\t" + str(loss.item()) + "\n"
		print(tmploss)
		
		trainloss_log += str(loss.item()) + "\n"
	return trainloss_log

# Training DNN...
def minibatch_train(model, optimizer, criterion, x, y, nepoch, batchsize):
	model.train()
	ntrain = len(x)
	trainloss_log = ""
	
	# training
	for epoch in tqdm(range(1, nepoch+1)):
		perm = np.random.permutation(ntrain)
		sumloss = 0
		for i in range(0, ntrain, batchsize):
			if i+batchsize > ntrain:
				size = ntrain - i
			else:
				size = batchsize
			xbatch = x[perm[i:i+size]]
			ybatch = y[perm[i:i+size]]
			
			pred_y = model(Variable(x))
			loss = criterion(pred_y, Variable(y))
			sumloss += loss * len(xbatch)
			# Reser optimizer state
			optimizer.zero_grad()
			
			# backpropagation
			loss.backward()
			optimizer.step()
		
		# standard output loss
		epochloss = sumloss / ntrain
		tmploss = str(epoch) + "epoch" + "\t" + str(epochloss.item()) + "\n"
		print(tmploss)
		
		trainloss_log += str(loss.item()) + "\n"
	return trainloss_log


if __name__ == "__main__":
	args = docopt(__doc__)
	print("Command line args:\n", args)
	filtersize = int(args['-f'])
	stride = int(args['-s'])
	padsize = int(args['-p'])
	numchannel = int(args['-c'])
	nepoch = int(args['-e'])
	batchsize = args['-b']
	outlosslog = args['--log']
	outmodel = args['-m']
	gpuid = int(args['-g'])
	
	dtype = torch.float
	device = torch.device("cuda:"+str(gpuid) if gpuid>=0 else "cpu")
	
	FFTSIZE = 1024
	FS = 16000 # [Hz]
	
	f02sp = F02SP(FFTSIZE,FS)
	
	f0 = 0.1 * np.arange(400,5000+1) # input, 0.1~800 [Hz]
	sp, discrete_freq = f02sp.get_sp(f0)
	input_sequence = discrete_freq/f0[:,np.newaxis]
	input_sequence = torch.from_numpy(input_sequence[:,np.newaxis]).to(dtype).to(device)
	sp = torch.from_numpy(sp[:,np.newaxis]).to(dtype).to(device)
	
	# model configure
	model = CNN(filtersize=filtersize, stride=stride, padsize=padsize, numchannel=numchannel)
	model = model.to(device)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters())
	
	# train
	if batchsize is None:
		losslog = batch_train(model, optimizer, criterion, x=input_sequence, y=sp, nepoch=nepoch)
	else:
		losslog = minibatch_train(model, optimizer, criterion, x=input_sequence, y=sp, nepoch=nepoch, batchsize=int(batchsize))
	
	# loss log save
	with open(outlosslog, mode="w") as f:
		f.write(losslog)
	
	# model save
	torch.save(model.state_dict(), outmodel)
	sys.exit(0)

