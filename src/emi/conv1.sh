#!/bin/tcsh
#BSUB -n 4
#BSUB -R span[ptile=1]
#BSUB -J Convolution[1-10]
#BSUB -W 17:30
#BSUB -oo /share/hmmrs/eplanch/NLP/results.%J_%I
#BSUB -e /share/hmmrs/eplanch/NLP/errors.%J_%I

module load julia
setenv JULIA_DEPOT_PATH /usr/local/usrapps/hmmrs/mlovig/julia
julia synth_convolution.jl $LSB_JOBINDEX
