"""Convert spectrogram to speech waveform.
usage: spec2speech.py [-h, --help] (--power <power_spfile>) (--phase <phase_spfile>) (-f <f0>) (-o <outwav>)

options:
  -h, --help                Show this message and exit.
  --power <power_spfile>    Power spectrum file (required).
  --phase <phase_spfile>    Phase spectrum file (required).
  -f <f0>                   F0 in Hz (required).
  -o <outwav>               Output wav file (required).
"""
from docopt import docopt

import numpy as np
from scipy.io import wavfile

import matplotlib.pyplot as plt
import sys

def istft(spectrum, f0, fp, fs):
	# spectrum:
	# fp: frame period [ms]
	# fs: sampling frequency [Hz]
	
	numframe, half_fftsize = spectrum.shape
	fftsize = 2 * (half_fftsize - 1)
	
	# ------------------- Mirroring ---------------------
	mirrored_spec = np.zeros((numframe,fftsize), dtype=np.complex64)
	mirrored_spec[:, :fftsize//2+1] = spectrum
	for i in range(1, fftsize//2):
		mirrored_spec[:, fftsize//2+i] = spectrum[:, fftsize//2-i].conjugate()
	# ---------------------------------------------------
	
	half_window_length = ((2.5 / f0 * fs) / 2).astype(np.int16)
	window_length = 2 * half_window_length
	
	step = int(fs * fp / 1000.)
	x_length = (numframe-1) * step + 1
	x = np.zeros(x_length, dtype=np.float32)
	wsum = np.zeros(x_length, dtype=np.float32)
	
	for frame in range(numframe):
		start = frame * step - half_window_length[frame]
		lower = -start if start < 0 else 0
		upper = x_length - 1 - start if x_length - 1 - start < window_length[frame] else window_length[frame]
		x[lower+start:start+upper] += np.fft.ifft(mirrored_spec[frame]).real[lower:upper]
		# z = np.zeros(x_length, dtype=np.float32)
		# z[lower+start:start+upper] = np.fft.ifft(mirrored_spec[frame]).real[lower:upper]
		# plt.plot(z)
		blackman_window = np.blackman(window_length[frame]).astype(np.float32)
		wsum[lower+start:start+upper] += blackman_window[lower:upper]
	# plt.show()
	pos = (wsum != 0)
	x[pos] /= wsum[pos]
	return x, wsum


if __name__ == '__main__':
	args = docopt(__doc__)
	print("Command line args:\n", args)
	power_spfile = args['--power']
	phase_spfile = args['--phase']
	F0 = float(args['-f'])
	outwav = args['-o']
	
	FFTSIZE = 1024
	
	power_sp = np.fromfile(power_spfile, dtype=np.float32, sep="").reshape(-1, FFTSIZE//2+1)
	phase_sp = np.fromfile(phase_spfile, dtype=np.float32, sep="").reshape(-1, FFTSIZE//2+1)
	f0 = np.array([F0] * power_sp.shape[0], dtype=np.float32)
	spectrum = np.sqrt(power_sp) * np.exp(1j * phase_sp)
	x, wsum = istft(spectrum, f0, fp=5.0, fs=16000)
	wavfile.write(outwav, 16000, x.astype(np.int16))
	sys.exit(0)
