#!/bin/tcsh
#BSUB -n 4
#BSUB -R span[ptile=1]
#BSUB -J real[1-50]
#BSUB -W 17:30
#BSUB -oo /share/hmmrs/eplanch/real_file/results.%J_%I
#BSUB -e /share/hmmrs/eplanch/real_file/errors.%J_%I

module load julia
setenv JULIA_DEPOT_PATH /usr/local/usrapps/hmmrs/mlovig/julia
julia convolution_real_data.jl $LSB_JOBINDEX
