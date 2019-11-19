#!/bin/tcsh -f

set sp = ../data/sp/mhtsda01.sp
set ap = ../data/ap/mhtsda01.ap

mkdir -p ../out/comp/
set outfig = ../out/comp/mhtsda01.eps

python -i ../tool/run.py -s $sp -a $ap -o $outfig
