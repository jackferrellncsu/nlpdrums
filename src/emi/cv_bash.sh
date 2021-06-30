#!/bin/tcsh
#BSUB -n 4
#BSUB -R span[ptile=1]
#BSUB -J cross_validation[1-1000]
#BSUB -W 12:00
#BSUB -oo /share/hmmrs/eplanch/cross_valid/results.%J_%I
#BSUB -e /share/hmmrs/eplanch/cross_valid/errors.%J_%I

module load julia
setenv JULIA_DEPOT_PATH /usr/local/usrapps/hmmrs/mlovig/julia
julia cv_synth_data.jl $LSB_JOBINDEX
