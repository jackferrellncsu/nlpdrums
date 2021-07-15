#!/bin/tcsh
#BSUB -n 4
#BSUB -R "select[rtx2080 || gtx1080 || p100 || k20m]"
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=yes"
#BSUB -J test
#BSUB -W 24:00
#BSUB -oo /share/hmmrs/jferrel3/NLP/results.%J
#BSUB -e /share/hmmrs/jferrel3/NLP/errors.%J

module load cuda 
module load julia
setenv JULIA_DEPOT_PATH /usr/local/usrapps/hmmrs/mlovig/julia
julia parrellization.jl
