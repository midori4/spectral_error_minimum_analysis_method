#!/bin/tcsh -f

set speaker = mht
set subset = a

set if0dir = ../data/if0/reaper/$speaker/$subset

mkdir -p ../out/spectrogram/{power,fig,phase}/{min,zero}/$speaker/$subset/

foreach if0 ( $if0dir/*05.if0 )
	set sp = ../data/spec/reaper/$speaker/$subset/$if0:t:r.sp
	set ap = ../data/ap/reaper/$speaker/$subset/$if0:t:r.ap
	set natural_spfile = /disk/fs1/bigtmp/hayasaka/study/m_thesis/spectral_error_minimum_analysis_method/01_tandem_spectrogram/out/spectrogram/binary/natural/$speaker/$subset/$if0:t:r.sp
	
	set phasetype = zero
	set outpower = ../out/spectrogram/power/$phasetype/$speaker/$subset/$if0:t:r.power
	set outphase = ../out/spectrogram/phase/$phasetype/$speaker/$subset/$if0:t:r.phase
	set outfig = ../out/spectrogram/fig/$phasetype/$speaker/$subset/$if0:t:r.eps
	set outdir = ../out/spectrogram/comp/$phasetype/$speaker/$subset/$if0:t:r
	mkdir -p $outdir
	
	python ../tool/speechparam2spec.py -f $if0 -s $sp -a $ap -n $natural_spfile \
			--outpower $outpower --outphase $outphase --outfig $outfig --outdir $outdir
	
	set phasetype = min
	set outpower = ../out/spectrogram/power/$phasetype/$speaker/$subset/$if0:t:r.power
	set outphase = ../out/spectrogram/phase/$phasetype/$speaker/$subset/$if0:t:r.phase
	set outfig = ../out/spectrogram/fig/$phasetype/$speaker/$subset/$if0:t:r.eps
	set outdir = ../out/spectrogram/comp/$phasetype/$speaker/$subset/$if0:t:r
	mkdir -p $outdir
	
	#python ../tool/speechparam2spec.py -f $if0 -s $sp -a $ap -n $natural_spfile \
	# 		--outpower $outpower --outphase $outphase --outfig $outfig --outdir $outdir --minphase
end
