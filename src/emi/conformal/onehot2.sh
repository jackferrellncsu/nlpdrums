#!/bin/tcsh
#BSUB -n 1
#BSUB -R span[ptile=1]
#BSUB -J ONEHOT
#BSUB -W 72:00
#BSUB -oo /share/hmmrs/eplanch/onehot/results.%J
#BSUB -e /share/hmmrs/eplanch/onehot/errors.%J

module load julia
setenv JULIA_DEPOT_PATH /usr/local/usrapps/hmmrs/mlovig/julia
julia onehot_clust2.jl $LSB_JOBINDEX
