"""
usage: train.py [-h|--help] (-l <numlayer>) (-u <numunit>) (-m <model_path>) (-g <gpuid>)

options:
  -h --help                 Show this message and exit.
  -l <numlayer>             The number of layers (required).
  -u <numunit>              The number of units (required).
  -m <model_path>           Trained model parameters (required).
  -g <gpuid>                GPU id, if it is less than 0 this processing executed on CPU.
"""
from docopt import docopt

import numpy as np
import torch

from model import MLP

import matplotlib.pyplot as plt


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
		return sp, self.discrete_freq


if __name__ == '__main__':
	args = docopt(__doc__)
	print("Command line args:\n", args)
	numlayer = int(args['-l'])
	numunit = int(args['-u'])
	model_path = args['-m']
	gpuid = int(args['-g'])
	
	dtype = torch.float
	device = torch.device("cuda:"+str(gpuid) if gpuid>=0 else "cpu")
	
	FFTSIZE = 1024
	FS = 16000 # [Hz]
	
	model = MLP(in_dim=FFTSIZE//2+1, out_dim=FFTSIZE//2+1, numlayer=numlayer, numunit=numunit)
	model.load_state_dict(torch.load(model_path))
	model = model.to(device)
	model.eval()
	
	f02sp = F02SP(FFTSIZE,FS)
	f0 = 0.1 * np.arange(200,5000+1) # input, 0.1~800 [Hz]
	sp, discrete_freq = f02sp.get_sp(f0)
	input_sequence = discrete_freq / f0[:,np.newaxis]
	input_sequence = torch.from_numpy(input_sequence).to(dtype).to(device)
	
	pred_sp = model(input_sequence)
	pred_sp = pred_sp.cpu().data.numpy()
