#!/bin/tcsh
#BSUB -n 4
#BSUB -R "select[rtx2080 || gtx1080]"
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=yes"
#BSUB -J onehot
#BSUB -W 24:00
#BSUB -oo /share/hmmrs/eplanch/onehot/results.%J
#BSUB -e /share/hmmrs/eplanch/onehot/errors.%J

module load cuda
module load julia
setenv JULIA_DEPOT_PATH /usr/local/usrapps/hmmrs/mlovig/julia
julia onehot_clust.jl $LSB_JOBINDEX
