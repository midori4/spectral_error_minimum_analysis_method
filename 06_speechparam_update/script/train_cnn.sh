#!/bin/tcsh -f

set filtersize = 3
set stride = 1
set padsize = 1
set numchannel = 8
set batchsize = 1
set nepoch = 100000

mkdir -p ../out/f02sp/

set outlosslog = ../out/f02sp/trainloss.txt
set outmodel = ../out/f02sp/model.pth

set gpuid = 3

python -i ../tool/train_cnn.py -f $filtersize -s $stride -p $padsize -c $numchannel -e $nepoch --log $outlosslog -m $outmodel -g $gpuid
