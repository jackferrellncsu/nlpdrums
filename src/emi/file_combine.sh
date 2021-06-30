#!/bin/tcsh
#BSUB -n 1
#BSUB -J Stich
#BSUB -W 00:30
#BSUB -oo /share/hmmrs/eplanch/real_file/results.%J
#BSUB -e /share/hmmrs/eplanch/real_file/errors.%J

module load julia
setenv JULIA_DEPOT_PATH /usr/local/usrapps/hmmrs/mlovig/julia
julia file_combine.jl
