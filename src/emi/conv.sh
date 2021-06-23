#!/bin/tcsh
#BSUB -n 4
#BSUB -W 17:30
#BSUB -oo results.%J
#BSUB -e errors.%J
#BSUB -J Convolution
module load julia
julia synth_convolution.jl
