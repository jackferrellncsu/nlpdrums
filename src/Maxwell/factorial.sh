#!/bin/tcsh 
#BSUB -n 4
#BSUB -W 0:01
#BSUB -oo results.%J
#BSUB -e errors.%J
#BSUB -J basicFactorial
module load julia
julia factorial.jl