#!/bin/tcsh
#BSUB -n 4
#BSUB -J nlpgroup
#BSUB -W 50:00
#BSUB -oo /share/hmmrs/jferrel3/NLP/results.%J
#BSUB -e /share/hmmrs/jferrel3/NLP/errors.%J


module load julia
setenv JULIA_DEPOT_PATH /share/hmmrs/jferrel3/julia
julia blstm_tensor_cpu.jl
