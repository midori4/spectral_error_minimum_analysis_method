#!/bin/tcsh -f

set numlayer = 1
set numunit = 1024

set model_path = ../out/f02sp/1layer_1024unit/model.pth

set gpuid = 1

python -i ../tool/test.py -l $numlayer -u $numunit -m $model_path -g $gpuid
