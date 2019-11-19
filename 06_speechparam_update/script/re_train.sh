#!/bin/tcsh -f

set numlayer = 1
set numunit = 8
set batchsize = 1
set nepoch = 10

mkdir -p ../out/f02sp/${numlayer}layer_${numunit}unit_${batchsize}batch

set outlosslog = ../out/f02sp/${numlayer}layer_${numunit}unit_${batchsize}batch/trainloss.txt
set outmodel = ../out/f02sp/${numlayer}layer_${numunit}unit_${batchsize}batch/model.pth

set gpuid = 2

python ../tool/re_train.py -l $numlayer -u $numunit -e $nepoch -b $batchsize --log $outlosslog -m $outmodel -g $gpuid
