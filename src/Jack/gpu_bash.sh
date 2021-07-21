#!/bin/tcsh
#BSUB -n 4
#BSUB -R "select[gtx1080]"
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=yes"
#BSUB -J nlpgroup
#BSUB -W 24:00
#BSUB -oo /share/hmmrs/jferrel3/NLP/results.%J
#BSUB -e /share/hmmrs/jferrel3/NLP/errors.%J

module load cuda/11.0
module load julia
setenv CUDA_VISIBLE_DEVICES = 3
setenv JULIA_DEPOT_PATH /usr/local/usrapps/hmmrs/jferrel3/julia
julia nextword_softmax_cluster.jl
