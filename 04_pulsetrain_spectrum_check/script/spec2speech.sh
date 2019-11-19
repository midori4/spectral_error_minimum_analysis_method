#!/bin/tcsh -f

set speaker = mht
set subset = a

set natural_spdir = /disk/fs1/bigtmp/hayasaka/study/m_thesis/spectral_error_minimum_analysis_method/01_tandem_spectrogram/out/spectrogram/binary/natural/$speaker/$subset/

foreach natural_sp ( $natural_spdir/*01.sp )
	set if0 = ../data/if0/reaper/mht/a/$natural_sp:t:r.if0
	set phase_sp = /disk/fs1/bigtmp/hayasaka/study/m_thesis/spectral_error_minimum_analysis_method/01_tandem_spectrogram/out/spectrogram/phase1/natural/$speaker/$subset/$natural_sp:t:r.phase

	# ---------------- natural speech resyn --------------------------------
	mkdir -p ../out/wav/natural/$speaker/$subset/
	set outwav = ../out/wav/natural/$speaker/$subset/$natural_sp:t:r.wav
	python ../tool/spec2speech.py --power $natural_sp --phase $phase_sp -f $if0 -o $outwav
	
	# ---------------- vocal tract filter: zero phase ----------------------
	set phasetype = zero
	set power_sp = ../out/spectrogram/power/$phasetype/$speaker/$subset/$natural_sp:t:r.power
	mkdir -p ../out/wav/synthesis/$phasetype/$speaker/$subset/
	set outwav = ../out/wav/synthesis/$phasetype/$speaker/$subset/$natural_sp:t:r.wav
	python ../tool/spec2speech.py --power $power_sp --phase $phase_sp -f $if0 -o $outwav
	
	# ---------------- vocal tract filter: minimum phase -------------------
	set phasetype = min
	set power_sp = ../out/spectrogram/power/$phasetype/$speaker/$subset/$natural_sp:t:r.power
	mkdir -p ../out/wav/synthesis/$phasetype/$speaker/$subset/
	set outwav = ../out/wav/synthesis/$phasetype/$speaker/$subset/$natural_sp:t:r.wav
	#python ../tool/spec2speech.py --power $power_sp --phase $phase_sp -f $if0 -o $outwav
end

