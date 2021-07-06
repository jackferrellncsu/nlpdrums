#!/bin/tcsh
#BSUB -n 4
#BSUB -R span[ptile=1]
#BSUB -J DTM_create[1-10]
#BSUB -W 2:00
#BSUB -oo /share/hmmrs/eplanch/DTM/results.%J_%I
#BSUB -e /share/hmmrs/eplanch/DTM/errors.%J_%I

module load julia
setenv JULIA_DEPOT_PATH /usr/local/usrapps/hmmrs/mlovig/julia
julia dtm_cluster.jl $LSB_JOBINDEX
