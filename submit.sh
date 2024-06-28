#!/bin/bash
#BSUB -n 1
#BSUB -W 300
#BSUB -q gpu
#BSUB -m "gpu_h100 gpu_a100"
#BSUB -gpu "num=1"
/usr/local/usrapps/bjreich/hyang23/PINN/bin/python /share/bjreich/hyang23/PINN/4_RBF_Matern.py

