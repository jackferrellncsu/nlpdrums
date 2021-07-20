#!/bin/tcsh
#BSUB -n 1
#BSUB -R "select[gtx1080]"
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=yes"
#BSUB -W 30

module load julia
module load cuda/11.0
setenv JULIA_DEPOT_PATH /usr/local/usrapps/hmmrs/jferrel3/julia

#bsub -Is -n 1 -R"select[gtx1080]" -q gpu -gpu "num=1:mode=shared:mps=yes" -W 30 tcsh
