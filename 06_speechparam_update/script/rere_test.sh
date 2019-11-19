#!/bin/tcsh -f

set numlayer = 1
set numunit = 1024

set model_path = ../out/f02sp/rere/1layer_1024unit_1batch/model.pth

set gpuid = 3

python -i ../tool/rere_test.py -l $numlayer -u $numunit -m $model_path -g $gpuid
