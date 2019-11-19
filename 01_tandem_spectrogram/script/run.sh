#!/bin/tcsh -f

set sp = mht
set set = a

set wavdir = ../../../../../corpus/ATR/B-set/wav/16k/$sp/sd/$set/

set type = natural
mkdir -p ../out/spectrogram/{fig,binary,phase1,phase2,power1,power2}/$type/$sp/$set/

foreach wav ( $wavdir/*.wav )
	set f0 = ../../../WORLD_anasyn_demo/out/if0/reaper/$sp/$set/$wav:t:r.if0
	set outfig = ../out/spectrogram/fig/$type/$sp/$set/$wav:t:r.eps
	set outsp = ../out/spectrogram/binary/$type/$sp/$set/$wav:t:r.sp
	set outphase1 = ../out/spectrogram/phase1/$type/$sp/$set/$wav:t:r.phase
	set outphase2 = ../out/spectrogram/phase2/$type/$sp/$set/$wav:t:r.phase
	set outpower1 = ../out/spectrogram/power1/$type/$sp/$set/$wav:t:r.power
	set outpower2 = ../out/spectrogram/power2/$type/$sp/$set/$wav:t:r.power
	python ../tool/run.py -i $wav -f $f0 --outfig $outfig --outsp $outsp \
			--outpower1 $outpower1 --outpower2 $outpower2 --outphase1 $outphase1 --outphase2 $outphase2
end

