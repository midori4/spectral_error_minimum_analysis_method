"""
usage: run.py [-h, --help] (-s <spectral_envelope>) (-a <aperiodicity>) (-o <outfig>)

options:
  -h, --help                    Show this message and exit.
  -s <spectral_envelope>        Spectral envelope (required).
  -a <aperiodicity>             Aperiodicity (required).
  -o <outfig>                   Output figure file (required).
"""
from docopt import docopt

import numpy as np
import matplotlib.pyplot as plt

class F02SP():
	def __init__(self, fftsize, fs):
		self.fftsize = fftsize
		self.N = self.fftsize // 2 + 1
		self.fs = fs
		self.nyquist_freq = fs // 2
		self.discrete_freq = self.nyquist_freq / (self.N - 1) * np.arange(self.N)

	def get_sp(self, f0):
		n = int(self.nyquist_freq / f0[0]) + 1
		sp = np.zeros(self.N, dtype=np.float32)
		for i in range(n):
			idx = np.abs(self.discrete_freq - i * f0[0]).argmin()
			sp[idx] = self.fftsize / np.sqrt(1 + self.fs / f0[0])
		# ------------------- Mirroring ---------------------
		mirrored_sp = np.zeros(self.fftsize, dtype=np.float32)
		mirrored_sp[:self.fftsize//2+1] = sp
		for i in range(1, self.fftsize//2):
			mirrored_sp[self.fftsize//2+i] = sp[self.fftsize//2-i]
		# ---------------------------------------------------
		return mirrored_sp

def get_minphase_spec(spenv):
	"""Convert (power) spectral envelope to minimum phase spectrum
	
	arguments:
	  spenv             with shape (numframe, fftsize//2+1)

	return:
	  minphase_sp       Complex array with shape (numframe, fftsize)
	"""
	MYSAFEGUARDMINIMUM = 1e-12
	numframe, half_fftsize = spenv.shape
	fftsize = 2 * (half_fftsize - 1)
	
	# ------------------- Mirroring ---------------------
	mirrored_spec = np.zeros((numframe,fftsize), dtype=np.float32)
	mirrored_spec[:, :fftsize//2+1] = spenv
	for i in range(1, fftsize//2):
		mirrored_spec[:, fftsize//2+i] = spenv[:, fftsize//2-i]
	# ---------------------------------------------------
	
	cepstrum = np.fft.ifft(np.log(mirrored_spec + MYSAFEGUARDMINIMUM)/2)
	
	# -------------- weighting -------------------
	cepstrum[:, 1:half_fftsize-1] *= 2.
	cepstrum[:, half_fftsize:] *= 0.
	
	minphase_spec = np.exp(np.fft.fft(cepstrum))
	return minphase_spec.astype(np.complex64)

def get_zerophase_spec(spenv):
	"""Convert (power) spectral envelope to zero phase spectrum
	
	arguments:
	  spenv             with shape (numframe, fftsize//2+1)

	return:
	  zerophase_sp      Complex array with shape (numframe, fftsize)
	"""
	numframe, half_fftsize = spenv.shape
	fftsize = 2 * (half_fftsize - 1)
	
	# ------------------- Mirroring ---------------------
	mirrored_spec = np.zeros((numframe,fftsize), dtype=np.float32)
	mirrored_spec[:, :fftsize//2+1] = spenv
	for i in range(1, fftsize//2):
		mirrored_spec[:, fftsize//2+i] = spenv[:, fftsize//2-i]
	# ---------------------------------------------------

	zerophase_spec = np.sqrt(mirrored_spec)
	return zerophase_spec

def get_randomphase_spec(spenv):
	"""Convert (power) spectral envelope to random phase spectrum
	
	arguments:
	  spenv             with shape (numframe, fftsize//2+1)

	return:
	  randomphase_sp    Complex array with shape (numframe, fftsize)
	"""
	numframe, half_fftsize = spenv.shape
	fftsize = 2 * (half_fftsize - 1)
	
	# ------------------- Mirroring ---------------------
	mirrored_spec = np.zeros((numframe,fftsize), dtype=np.float32)
	mirrored_spec[:, :fftsize//2+1] = spenv
	for i in range(1, fftsize//2):
		mirrored_spec[:, fftsize//2+i] = spenv[:, fftsize//2-i]
	# ---------------------------------------------------
	
	randomphase = 2j * np.pi * np.random.rand(numframe,fftsize)
	zerophase_spec = np.sqrt(mirrored_spec)*np.exp(randomphase)
	return zerophase_spec

# circular convolution
def spec_convolve(X, Y, fftsize):
	Z = np.zeros(fftsize, dtype=np.complex128)
	for k in range(fftsize):
		for p in range(fftsize):
			Z[k] += X[p] * Y[k-p]
	return Z / fftsize


if __name__ == '__main__':
	args = docopt(__doc__)
	print("Command line args:\n", args)
	sp_file = args['-s']
	ap_file = args['-a']
	outfig = args['-o']
	
	FFTSIZE = 1024
	FS = 16000 # [Hz]
	F0 = np.array([125], dtype=np.float32) # [Hz]
	
	f02sp = F02SP(FFTSIZE, FS)
	pulsetrain_sp = f02sp.get_sp(F0)
	
	sp = np.fromfile(sp_file, dtype=np.float32, sep="").reshape(-1, FFTSIZE//2+1)
	ap = np.fromfile(ap_file, dtype=np.float32, sep="").reshape(-1, FFTSIZE//2+1)
	
	FRAME = 100
	periodic_spenv = sp * (1 - ap**2)
	minphase_sp = get_minphase_spec(periodic_spenv)[FRAME]
	zerophase_sp = get_zerophase_spec(periodic_spenv)[FRAME]
	randomphase_sp = get_randomphase_spec(periodic_spenv)[FRAME]
	
	minphase_spec = pulsetrain_sp * minphase_sp
	zerophase_spec = pulsetrain_sp * zerophase_sp
	randomphase_spec = pulsetrain_sp * randomphase_sp
	
	half_window_length = int((2.5 / F0[0] * FS) / 2)
	zeropad_size = FFTSIZE - half_window_length * 2
	blackman_window = np.pad(np.blackman(2 * half_window_length), [0,zeropad_size], 'constant')
	W = np.fft.fft(blackman_window)
	
# 	plt.plot(20*np.log10(np.abs(minphase_spec)[:FFTSIZE//2+1]+1e-12), linewidth=0.5, label='minphase')
# 	plt.plot(20*np.log10(np.abs(zerophase_spec)[:FFTSIZE//2+1]+1e-12), linewidth=0.5, label='zerophase')
# 	plt.plot(20*np.log10(np.abs(randomphase_spec)[:FFTSIZE//2+1]+1e-12), linewidth=0.5, label='randomphase')
	
	plt.plot(20*np.log10(np.abs(spec_convolve(W, minphase_spec, FFTSIZE)[:FFTSIZE//2+1])), linewidth=0.5, label='minphase')
	plt.plot(20*np.log10(np.abs(spec_convolve(W, zerophase_spec, FFTSIZE)[:FFTSIZE//2+1])), linewidth=0.5, label='zerophase')
	plt.plot(20*np.log10(np.abs(spec_convolve(W, randomphase_spec, FFTSIZE)[:FFTSIZE//2+1])), linewidth=0.5, label='randomphase')
	plt.legend()
	plt.savefig(outfig)
