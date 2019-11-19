#!/bin/tcsh -f

mkdir -p ../out/spectrum/{pulsetrain,whitenoise}/

set pulsepowersp = ../out/spectrum/pulsetrain/power.sp
set pulsephasesp = ../out/spectrum/pulsetrain/phase.sp

set noisepowersp = ../out/spectrum/whitenoise/power.sp
set noisephasesp = ../out/spectrum/whitenoise/phase.sp

python -i ../tool/run.py --outpower1 $pulsepowersp --outphase1 $pulsephasesp --outpower2 $noisepowersp --outphase2 $noisephasesp
