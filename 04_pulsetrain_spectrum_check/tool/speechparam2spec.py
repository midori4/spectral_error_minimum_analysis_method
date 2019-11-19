"""Generate amplitude spectrum from speech parameter and synthesize speech waveform from it.
usage: speechparam2spec.py [-h, --help] (-f <f0_file>) (-s <spectrum_file>) (-a <aperiodicity_file>) (-n <natural_spfile>) (--outpower <outpower>) (--outphase <outphase>) (--outfig <outfig>) (--outdir <outdir>) [--minphase] 

options:
  -h, --help                    Show this message and exit.
  -f <f0_file>                  F0 file name (required).
  -s <spectrum_file>            Spectral envelope file name (required).
  -a <aperiodicity_file>        Aperiodicity file name (required).
  -n <natural_spfile>           Spectrogram file of natural speech (required).
  --outpower <outpower>         Spectrogram save file name in binary format (required).
  --outphase <outphase>         Phase spectrum save file namr in binary format (required).
  --outfig <outfig>             Spectrogram save file name in eps format (required).
  --outdir <outdir>             Out directory for comparison between generated and natural spectrum (required).
  --minphase                    If specified, minimum phase used as phase of vocal tract fileter (optional).
"""
from docopt import docopt

import numpy as np
from os.path import join

import matplotlib.pyplot as plt

import time
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
				sp[frame][idx] = self.fftsize / np.sqrt(1 + self.fs / f0[frame])
		return sp

def get_minphase_spec(spectrogram):
	"""Convert (power) spectrogram to minimum phase spectrum
	
	arguments:
	  spectrogram       with shape (numframe, fftsize//2+1)

	return:
	  minphase_sp       Complex array with shape (numframe, fftsize)
	"""
	numframe, half_fftsize = spectrogram.shape
	fftsize = 2 * (half_fftsize - 1)
	
	# ------------------- Mirroring ---------------------
	mirrored_spec = np.zeros((numframe,fftsize), dtype=np.float32)
	mirrored_spec[:, :fftsize//2+1] = spectrogram
	for i in range(1, fftsize//2):
		mirrored_spec[:, fftsize//2+i] = spectrogram[:, fftsize//2-i]
	# ---------------------------------------------------
	
	cepstrum = np.fft.ifft(np.log(mirrored_spec+1e-12)/2)
	
	# -------------- weighting -------------------
	cepstrum[:, 1:half_fftsize-1] *= 2.
	cepstrum[:, half_fftsize:] *= 0.
	
	minphase_spec = np.exp(np.fft.fft(cepstrum))
	return minphase_spec.astype(np.complex64)

def spec_convolve(X, Y, f0length, fftsize):
	X1 = np.fft.ifft(X).astype(np.complex64)
	Y1 = np.fft.ifft(Y).astype(np.complex64)
	Z = np.fft.fft(X1*Y1).astype(np.complex64)
	return Z

if __name__ == '__main__':
	args = docopt(__doc__)
	print("Command line args:\n", args)
	if0_file = args['-f']
	sp_file = args['-s']
	ap_file = args['-a']
	natural_spfile = args['-n']
	outpower = args['--outpower']
	outphase = args['--outphase']
	outfig = args['--outfig']
	outdir = args['--outdir']
	minphase = args['--minphase']
	
	FFTSIZE = 1024
	FS = 16000 # [Hz]
	FRAMEPERIOD = 5.0 # [ms]
	
	if0 = np.fromfile(if0_file, dtype=np.float32, sep="").reshape(-1)
	F0LENGTH = len(if0)
	sp = np.fromfile(sp_file, dtype=np.float32, sep="").reshape(-1, FFTSIZE//2+1)
	ap = np.fromfile(ap_file, dtype=np.float32, sep="").reshape(-1, FFTSIZE//2+1)
	
	# -------------------- white noise generation ---------------------
	white_noise = np.random.normal(loc=0., scale=1., size=FFTSIZE).astype(np.float32)
	white_noise_sp = np.fft.fft(white_noise)[:FFTSIZE//2+1]
	
	#phase = 2 * np.pi * np.random.rand(FFTSIZE//2+1).astype(np.float32)
	#white_noise_sp = np.sqrt(FFTSIZE) * np.exp(1j*phase)
	
	# normalize failed ↓  
	# white_noise_sp = np.sqrt(FFTSIZE) * np.ones(FFTSIZE//2+1, dtype=np.float32)
	# -----------------------------------------------------------------
	
	f02sp = F02SP(FFTSIZE, FS)
	pulse_train_sp = f02sp.get_sp(if0)
	if minphase:
		periodic_spenv = get_minphase_spec((1 - ap**2) * sp)[:,:FFTSIZE//2+1]
		aperiodic_spenv = get_minphase_spec(ap**2 * sp)[:,:FFTSIZE//2+1]
	else:
		periodic_spenv = np.sqrt((1 - ap**2) * sp)
		aperiodic_spenv = np.sqrt(ap**2 * sp)
	spec = pulse_train_sp * periodic_spenv + white_noise_sp * aperiodic_spenv
	mirrored_spec = np.zeros((F0LENGTH,FFTSIZE), dtype=np.complex64)
	mirrored_spec[:,:FFTSIZE//2+1] = spec
	for i in range(1, FFTSIZE//2):
		mirrored_spec[:,FFTSIZE//2+i] = spec[:,FFTSIZE//2-i].conjugate()
	
	window_sp = np.zeros((F0LENGTH,FFTSIZE), dtype=np.complex64)
	for i in range(F0LENGTH):
		current_f0 = if0[i]
		analysis_time = i * FRAMEPERIOD / 1000. # [s]
		half_window_length = int((2.5 / current_f0 * FS) / 2)
		window_length = 2 * half_window_length + 1
		zeropad = FFTSIZE - window_length
		blackman_window = np.pad(np.blackman(window_length), [0,zeropad], 'constant')
		W = np.fft.fft(blackman_window)
		window_sp[i] = W
		
	t1 = time.time()
	spectrum = spec_convolve(window_sp, mirrored_spec, F0LENGTH, FFTSIZE)[:,:FFTSIZE//2+1]
	t2 = time.time()
	elapsed_time = t2 - t1
	print("Elapsed time:{}".format(elapsed_time))
	
	phase_sp = np.angle(spectrum).astype(np.float32)
	phase_sp.tofile(outphase)
	power_sp = np.abs(spectrum, dtype=np.float32)**2
	power_sp.tofile(outpower)
	
	plt.imshow(10*np.log10(power_sp[:,::-1]).T, extent=[0, F0LENGTH*FRAMEPERIOD/1000., 0, FS/2], aspect="auto")
	plt.xlabel('Time [s]')
	plt.ylabel('Frequency [Hz]')
	plt.savefig(outfig)
	plt.close()
	
	natural_sp = np.fromfile(natural_spfile, dtype=np.float32, sep="").reshape(-1, FFTSIZE//2+1)
	for i in range(int(F0LENGTH // 100) + 1):
	# for i in range(1,2):
		plt.plot(10*np.log10(natural_sp[i*100]), label='natural')
		plt.plot(10*np.log10(power_sp[i*100]), label='from speech param')
		plt.legend()
		plt.savefig(join(outdir,'{0:04d}'.format(i*100)+'frame.png'))
		plt.close()
	
