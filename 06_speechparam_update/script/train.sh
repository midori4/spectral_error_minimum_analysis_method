#!/bin/tcsh -f

set numlayer = 3
set numunit = 1024
set nepoch = 100000

mkdir -p ../out/f02sp/${numlayer}layer_${numunit}unit

set outlosslog = ../out/f02sp/${numlayer}layer_${numunit}unit/trainloss.txt
set outmodel = ../out/f02sp/${numlayer}layer_${numunit}unit/model.pth

set gpuid = 0

python ../tool/train.py -l $numlayer -u $numunit -e $nepoch --log $outlosslog -m $outmodel -g $gpuid

# set batchsize = 1
# python ../tool/train.py -l $numlayer -u $numunit -e $nepoch -b $batchsize --log $outlosslog -m $outmodel -g $gpuid


