bsub -Is -n 1 -R"select[rtx2080 || gtx1080]" -q gpu -gpu "num=1:mode=shared:mps=yes" -W 10 tcsh
