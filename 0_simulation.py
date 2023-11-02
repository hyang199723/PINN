# Generate multi time step data necessary for experiments
# Simulate from two non-stationary process: A vertically more correlated process
# and local stationary process

#%% Packages
import sys
# wk_dir = "/Users/hongjianyang/PINN_SPDE/"
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/PINN/"
sys.path.append(wk_dir)
from spde import *
import numpy as np
import pandas as pd
#%% Vertical more correlated
rho = 3
vvv = 1
N = 200
ts = 10 # 10 time steps

X, Y = gen_non_same(N, rho, vvv)
X = pd.DataFrame(X)
Y = pd.DataFrame(Y)
data = pd.concat([X, Y], axis = 1)

data.to_csv("non_stat_200.csv")
