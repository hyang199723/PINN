#!/bin/bash
#BSUB -n 1
#BSUB -W 400
#BSUB -q stat_gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -o out.%J
#BSUB -e err.%J
module load PrgEnv-pgi
module load cuda
/usr/local/usrapps/bjreich/hyang23/PINN /share/bjreich/hyang23/PINN/4_RBF_Matern.py
