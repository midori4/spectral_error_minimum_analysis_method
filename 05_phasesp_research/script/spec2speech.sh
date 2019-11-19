#!/bin/tcsh -f

set F0 = 100.

mkdir -p ../out/wav/

set power_sp = ../out/spectrum/pulsetrain/power.sp
set phase_sp = ../out/spectrum/pulsetrain/phase.sp
set outwav = ../out/wav/pulsepower_pulsephase.wav
python ../tool/spec2speech.py --power $power_sp --phase $phase_sp -f $F0 -o $outwav
	
set power_sp = ../out/spectrum/pulsetrain/power.sp
set phase_sp = ../out/spectrum/whitenoise/phase.sp
set outwav = ../out/wav/pulsepower_noisephase.wav
python ../tool/spec2speech.py --power $power_sp --phase $phase_sp -f $F0 -o $outwav
	

set power_sp = ../out/spectrum/whitenoise/power.sp
set phase_sp = ../out/spectrum/pulsetrain/phase.sp
set outwav = ../out/wav/noisepower_pulsephase.wav
python ../tool/spec2speech.py --power $power_sp --phase $phase_sp -f $F0 -o $outwav

set power_sp = ../out/spectrum/whitenoise/power.sp
set phase_sp = ../out/spectrum/whitenoise/phase.sp
set outwav = ../out/wav/noisepower_noisephase.wav
python ../tool/spec2speech.py --power $power_sp --phase $phase_sp -f $F0 -o $outwav

