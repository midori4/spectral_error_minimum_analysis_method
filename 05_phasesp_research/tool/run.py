"""TANDEM spectrum calculation of pulse train and white noise.

usage: run.py [-h|--help] (--outpower1 <pulsepowersp>) (--outphase1 <pulsephasesp>) (--outpower2 <noisepowersp>) (--outphase2 <noisephasesp>)

options:
  -h --help                     Show this message and exit.
  --outpower1 <pulsepowersp>    Power spectrum file of pulse train (required).
  --outphase1 <pulsephasesp>    Phase spectrum file of pulse train (required).
  --outpower2 <noisepowersp>    Power spectrum file of white noise (required).
  --outphase2 <noisephasesp>    Phase spectrum file of white noise (required).
"""
from docopt import docopt

import numpy as np
import sys

def pulsetrain_including_noise(f0, fs, duration, ratio):
	xlength = int(duration * fs)
	time = np.arange(xlength) / fs
	x = np.zeros(xlength, dtype=np.float32)
	pulsetrain = np.zeros(xlength, dtype=np.float32)
	numpulse = 0
	while time[-1] > numpulse / f0:
		pos = np.abs(time - numpulse / f0).argmin()
		pulsetrain[pos] = 1000. * np.sqrt(fs / f0)
		numpulse += 1
	whitenoise = np.sqrt(1000.) * np.random.normal(loc=0., scale=1., size=xlength).astype(np.float32)
	x = np.sqrt((1 - ratio) * pulsetrain**2 + ratio * whitenoise**2)
	return x

def tandem_spectrum(x, f0, frame_priod, fs, fftsize):
	xlength = len(x)
	half_window_length = int((2.5 / f0 * fs) / 2)
	window_length = 2 * half_window_length
	blackman_window = np.blackman(window_length).astype(np.float32)
	
	numframe = int(xlength / (frame_period / 1000 * fs))
	power_sp = np.zeros((numframe,fftsize//2+1), dtype=np.float32)
	phase_sp = np.zeros((numframe,fftsize//2+1), dtype=np.float32)
	for frame in range(numframe):
		current_time = frame * frame_period / 1000 #[s]
		t1 = int((current_time - 1 / f0 / 4) * fs)
		t2 = int((current_time + 1 / f0 / 4) * fs)
		start1 = t1 - half_window_length
		start2 = t2 - half_window_length
		
		# ----------------- spectrogram calculation --------------------------
		if start1 < 0:
			lower = -start1
			windowed_speech1 = np.pad(x[:window_length-lower], [lower,0], 'edge').copy()
		elif start1+window_length > xlength:
			upper = (start1 + window_length) - xlength
			windowed_speech1 = np.pad(x[start1:], [0, upper], 'edge').copy()
		else:
			windowed_speech1 = x[start1:start1+window_length].copy()
		
		if start2 < 0:
			lower = -start2
			windowed_speech2 = np.pad(x[:window_length-lower], [lower,0], 'edge').copy()
		elif start2+window_length > xlength:
			upper = (start2 + window_length) - xlength
			windowed_speech2 = np.pad(x[start2:], [0, upper], 'edge').copy()
		else:
			windowed_speech2 = x[start2:start2+window_length].copy()
		
		windowed_speech1 *= blackman_window
		windowed_speech2 *= blackman_window
		
		diff = fftsize - window_length
		windowed_speech1_for_fft = np.pad(windowed_speech1, [0,diff], 'constant')
		windowed_speech2_for_fft = np.pad(windowed_speech2, [0,diff], 'constant')
		
		sp1 = np.fft.fft(windowed_speech1_for_fft)[:fftsize//2+1]
		sp2 = np.fft.fft(windowed_speech2_for_fft)[:fftsize//2+1]
		
		phase_sp[frame] = np.angle(sp1)
		power_sp[frame] = (np.abs(sp1)**2 + np.abs(sp2)**2) / 2
	return power_sp, phase_sp
	

if __name__ == '__main__':
	args = docopt(__doc__)
	print("Command line args:\n", args)
	pulsetrain_powersp = args['--outpower1']
	pulsetrain_phasesp = args['--outphase1']
	noise_powersp = args['--outpower2']
	noise_phasesp = args['--outphase2']

	F0 = 100. # [Hz]
	FS = 16000 # sampling frequency [Hz]
	DURATION = 1. # [s]
	
	pulsetrain = pulsetrain_including_noise(F0, FS, DURATION, ratio=0.)
	whitenoise = np.sqrt(1000.) * np.random.normal(loc=0., scale=1., size=len(pulsetrain)).astype(np.float32)
	
	frame_period = 5.0 # [ms]
	power_sp, phase_sp = tandem_spectrum(pulsetrain, F0, frame_period, FS, fftsize=1024)
	power_sp.tofile(pulsetrain_powersp)
	phase_sp.tofile(pulsetrain_phasesp)
	
	power_sp, phase_sp = tandem_spectrum(whitenoise, F0, frame_period, FS, fftsize=1024)
	power_sp.tofile(noise_powersp)
	phase_sp.tofile(noise_phasesp)
	sys.exit(0)
