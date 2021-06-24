#!/bin/tcsh
#BSUB -n 4
#BSUB -R span[ptile=1]
#BSUB -J Glove[1-1000]
#BSUB -W 17:30
#BSUB -oo /share/hmmrs/jferrel3/NLP/results.%J
#BSUB -e /share/hmmrs/jferrel3/NLP/errors.%J

module load julia
setenv JULIA_DEPOT_PATH /usr/local/usrapps/hmmrs/mlovig/julia
julia glove_synth_cluster.jl $LSB_JOBINDEX
