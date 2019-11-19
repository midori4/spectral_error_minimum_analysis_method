"""Generate spectrogram using TANDEM-window.

usage: run.py [-h|--help] (-i <input_wav>) (-f <f0>) (--outfig <outfig>) (--outsp <outsp>) (--outpower1 <outpower1>) (--outpower2 <outpower2>) (--outphase1 <outphase1>) (--outphase2 <outphase2>)

options:
  -h --help                 Show this message and exit.
  -i <input_wav>            Wav file name (required).
  -f <f0>                   F0 file name (required).
  --outfig <outfig>         Spectrogram save file in eps format (required).
  --outsp <outsp>           Spectrogram save file in binary format (required).
  --outpower1 <outpower1>   Out half period before power spectrum (required).
  --outpower2 <outpower2>   Out half period after power spectrum (required).
  --outphase1 <outphase1>   Out half period before phase spectrum (required).
  --outphase2 <outphase2>   Out half period after phase spectrum (required).
"""
from docopt import docopt

import numpy as np

import pyworld
from scipy.io import wavfile

import matplotlib.pyplot as plt


if __name__ == '__main__':
	args = docopt(__doc__)
	print("Command line args:\n", args)
	inwav = args['-i']
	f0file = args['-f']
	outfig = args['--outfig']
	outsp = args['--outsp']
	outpower1 = args['--outpower1']
	outpower2 = args['--outpower2']
	outphase1 = args['--outphase1']
	outphase2 = args['--outphase2']
	
	frame_period = 5.0 # [ms]
	fft_length = 1024
	
	fs, wav_data = wavfile.read(inwav)
	wav_data = wav_data.astype(np.float32)
	wav_length = wav_data.shape[0]
	
	f0 = np.fromfile(f0file, dtype=np.float32, sep="").reshape(-1)
	num_frame = f0.shape[0]
	
	phase1 = np.zeros((num_frame, fft_length//2+1), dtype=np.float32)
	phase2 = np.zeros((num_frame, fft_length//2+1), dtype=np.float32)
	power1 = np.zeros((num_frame, fft_length//2+1), dtype=np.float32)
	power2 = np.zeros((num_frame, fft_length//2+1), dtype=np.float32)
	for n in range(num_frame):
		if f0[n] == 0.:
			current_f0 = 500.
		else:
			current_f0 = f0[n]
		
		half_window_length = int((2.5 / current_f0 * fs) / 2)
		window_length = 2 * half_window_length + 1
		blackman_window = np.blackman(window_length).astype(np.float32)
		
		current_time_sample = int((n * frame_period / 1000) * fs) # [sample]
		quarter_period_sample = int((1 / current_f0) / 4 * fs)
		t1 = current_time_sample - quarter_period_sample
		t2 = current_time_sample + quarter_period_sample
		start1 = t1 - half_window_length
		start2 = t2 - half_window_length
		
		# ----------------- spectrogram calculation --------------------------
		if start1 < 0:
			lower = -start1
			windowed_speech1 = np.pad(wav_data[:window_length-lower], [lower,0], 'edge').copy()
		elif start1+window_length > wav_length:
			upper = (start1 + window_length) - wav_length
			windowed_speech1 = np.pad(wav_data[start1:], [0, upper], 'edge').copy()
		else:
			windowed_speech1 = wav_data[start1:start1+window_length].copy()
		
		if start2 < 0:
			lower = -start2
			windowed_speech2 = np.pad(wav_data[:window_length-lower], [lower,0], 'edge').copy()
		elif start2+window_length > wav_length:
			upper = (start2 + window_length) - wav_length
			windowed_speech2 = np.pad(wav_data[start2:], [0, upper], 'edge').copy()
		else:
			windowed_speech2 = wav_data[start2:start2+window_length].copy()
		
		windowed_speech1 *= blackman_window
		windowed_speech2 *= blackman_window
		
		diff = fft_length - window_length
		windowed_speech1_for_fft = np.pad(windowed_speech1, [0,diff], 'constant')
		windowed_speech2_for_fft = np.pad(windowed_speech2, [0,diff], 'constant')
		
		# sp1 = (np.fft.fft(windowed_speech1_for_fft) * np.exp(2j*np.pi*(quarter_period_sample+half_window_length)/fft_length*np.arange(fft_length)))[:fft_length//2+1]
		# sp2 = (np.fft.fft(windowed_speech2_for_fft) * np.exp(-2j*np.pi*(quarter_period_sample-half_window_length)/fft_length*np.arange(fft_length)))[:fft_length//2+1]
		sp1 = np.fft.fft(windowed_speech1_for_fft)[:fft_length//2+1]
		sp2 = np.fft.fft(windowed_speech2_for_fft)[:fft_length//2+1]
		
		phase1[n] = np.angle(sp1)
		phase2[n] = np.angle(sp2)
		
		power1[n] = np.abs(sp1)**2
		power2[n] = np.abs(sp2)**2
		
	spectrogram = (power1 + power2) / 2 # average
	spectrogram.tofile(outsp)
	phase1.tofile(outphase1)
	phase2.tofile(outphase2)
	power1.tofile(outpower1)
	power2.tofile(outpower2)
	plt.imshow(10*np.log10(spectrogram[:,::-1]).T, extent=[0, num_frame*frame_period/1000., 0, fs/2], aspect="auto")
	plt.xlabel('Time [s]')
	plt.ylabel('Frequency [Hz]')
	plt.savefig(outfig)
	plt.close()
