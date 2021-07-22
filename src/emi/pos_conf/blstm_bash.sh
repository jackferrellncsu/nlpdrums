#!/bin/tcsh
#BSUB -n 1
#BSUB -J BLSTM
#BSUB -W 24:00
#BSUB -oo /share/hmmrs/eplanch/LSTM/results.%J
#BSUB -e /share/hmmrs/eplanch/LSTM/errors.%J

module load julia
setenv JULIA_DEPOT_PATH /usr/local/usrapps/hmmrs/mlovig/julia
julia blstm_clust.jl
