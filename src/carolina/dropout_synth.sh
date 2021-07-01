#!/bin/tcsh
#BSUB -n 4
#BSUB -R span[ptile=1]
#BSUB -J Glove[1-700]
#BSUB -W 17:30
#BSUB -oo /share/hmmrs/ckapper/results.%J
#BSUB -e /share/hmmrs/ckapper/errors.%J

module load julia
setenv JULIA_DEPOT_PATH /usr/local/usrapps/hmmrs/mlovig/julia
julia FFdropout_synth.jl $LSB_JOBINDEX
