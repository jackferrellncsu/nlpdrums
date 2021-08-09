#!/bin/sh
SEED=1

#Activate python env
source .venv/Scripts/activate

#Generates data into Data folder
python src/Code_Snapshot/RunFile_mlm_datapr.py -s $SEED

