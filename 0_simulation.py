# Simulate and save data of different spatial covariance structure

#%% Packages
import sys
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/PINN/"
sys.path.append(wk_dir)
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from spde import *
from scipy.special import gamma, kn
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')
import scipy.stats as stats
import pylab

# %% 2-D stationary process
N = 1300
P = 2
noise_var = 0.1
rho = 0.2
nu = 1
kappa = (8 * nu)**(0.5) / rho
spatial_var = 1

iters = 100
matern_data = np.zeros((N, 3, iters))
for i in range(100):
    X, Y = gen_matern(N, rho, spatial_var, noise_var, nu)
    X = X[:, 1:3] # Only need coors
    matern_data[:, 0:2, i] = X
    matern_data[:, 2, i] = Y
# Save to data file for comparison
matern_data_flat = matern_data.reshape(-1, matern_data.shape[-1])
np.savetxt("Data/matern_02_1_1.csv", matern_data_flat, delimiter=",")

# %% 2-D stationary process with higher spatial and nugget variance
N = 1300
P = 2
noise_var = 5
rho = 0.2
nu = 1
kappa = (8 * nu)**(0.5) / rho
spatial_var = 5

iters = 100
matern_data = np.zeros((N, 3, iters))
for i in range(100):
    X, Y = gen_matern(N, rho, spatial_var, noise_var, nu)
    X = X[:, 1:3] # Only need coors
    matern_data[:, 0:2, i] = X
    matern_data[:, 2, i] = Y
# Save to data file for comparison
matern_data_flat = matern_data.reshape(-1, matern_data.shape[-1])
np.savetxt("Data/matern_02_1_5.csv", matern_data_flat, delimiter=",")