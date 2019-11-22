#!/bin/tcsh -f

set outdir = ../out/2layer_1024unit
mkdir -p $outdir

python ../tool/train.py -o $outdir
