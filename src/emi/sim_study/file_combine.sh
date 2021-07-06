#!/bin/tcsh
#BSUB -n 1
#BSUB -J Combine
#BSUB -W 02:00
#BSUB -oo /share/hmmrs/eplanch/cross_valid/results.%J
#BSUB -e /share/hmmrs/eplanch/cross_valid/errors.%J

module load julia
setenv JULIA_DEPOT_PATH /usr/local/usrapps/hmmrs/mlovig/julia
julia file_combine.jl
