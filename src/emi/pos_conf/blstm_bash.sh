#!/bin/tcsh
#BSUB -n 4
#BSUB -R "select[gtx1080]"
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=yes"
#BSUB -J blstm_model
#BSUB -W 24:00
#BSUB -oo /share/hmmrs/eplanch/blstm/results.%J
#BSUB -e /share/hmmrs/eplanch/blstm/errors.%J

module load cuda/10.2
module load julia
setenv JULIA_DEPOT_PATH /share/hmmrs/jferrel3/julia
julia blstm_tensor_gpu.jl
