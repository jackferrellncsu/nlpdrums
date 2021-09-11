#!/bin/tcsh
#BSUB -n 4
#BSUB -J nlpgroup
#BSUB -W 24:00
#BSUB -oo /share/hmmrs/jferrel3/NLP/results.%J
#BSUB -e /share/hmmrs/jferrel3/NLP/errors.%J

module load conda
conda activate /share/hmmrs/jferrel3/envs
python RunFile_mlm_bert.py -s 1
