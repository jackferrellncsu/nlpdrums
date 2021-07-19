bsub -Is -n 1 -R"select[gtx1080]" -q gpu -gpu "num=1:mode=shared:mps=yes" -W 30 tcsh
module load julia
module load cuda
