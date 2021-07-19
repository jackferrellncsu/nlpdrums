#!/bin/tcsh
#BSUB -n 1
#BSUB -R span[ptile=1]
#BSUB -J lstm
#BSUB -W 20:00
#BSUB -oo /share/hmmrs/eplanch/LSTM/results.%J
#BSUB -e /share/hmmrs/eplanch/LSTM/errors.%J

module load julia
setenv JULIA_DEPOT_PATH /usr/local/usrapps/hmmrs/mlovig/julia
julia lstm_prediction.jl $LSB_JOBINDEX
