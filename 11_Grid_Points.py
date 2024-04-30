#%%  Generate data on grids and take derivative
import sys
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/PINN/"
sys.path.append(wk_dir)
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from scipy.special import gamma, kn
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')
import scipy.stats as stats
import pylab
from spde import *
#%% Generate data
coords = np.array([[i, j] for i in range(1, 36, 1) for j in range(1, 36, 1)]) / 36
noise = np.random.normal(0, 0.001, (1225, 2))
coords = coords + noise
def Matern_Cor(nu, rho, distance):
    kappa = (8 * nu)**(0.5) / rho
    const = 1 / (2**(nu - 1) * gamma(nu))
    kd = kappa * distance
    first_term = kd**nu
    second_term = kv(nu, kd)
    second_term[np.diag_indices_from(second_term)] = 0.
    out = const * first_term * second_term
    out[np.diag_indices_from(out)] = 1.0
    return out

n = 35**2
rho = 0.2
nu = 1
kappa = (8 * nu)**(0.5) / rho
spatial_var = 1
noise_std = 0.1

X = np.zeros((n, 3))
X[:, 0] = 1
X[:, 1:3] = coords

# Exponential Correlation
distance = distance_matrix(coords.reshape(-1, 2), coords.reshape(-1, 2))
corr = Matern_Cor(nu, rho, distance)
# Cholesky decomposition and generate correlated data
L = np.linalg.cholesky(spatial_var*corr)
z = np.random.normal(0, 1, n)
Y = np.dot(L, z) # + np.random.normal(0, noise_std, n) # Don't include random noise for now

#%% Visualize
plt.scatter(X[:,1], X[:,2], s = 20, c = Y)

#%% Run through NN and calculate W process
# basis function
num_basis = [3,5,6,8]
#num_basis = [7,9,10,12]
squared = [i**2 for i in num_basis]
out_dim = np.sum(squared)
fixed_centers = torch.randn(out_dim, 2)
sum = 0
for i in num_basis:
    loc = np.linspace(0,1,i)
    x = np.array([(x, y) for x in loc for y in loc])
    fixed_centers[sum:sum+i**2] = torch.tensor(x)
    sum += i**2

X_train = X_val = X[:, 1:3]
y_train = Y
lr = 0.003
nnn = 5000 # Numbr of discrete grid of points to evaluate kde
lower = -5
upper = 5
x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
theoretical_pdf = norm.pdf(x, 0, 1)

model_1, density, W = RBF_train(X_train, X_train, y_train, y_train, lr=lr, epochs=3500, alpha = 0,
                          device = device, centers=fixed_centers, dims = out_dim, theory = theoretical_pdf)

www = W.cpu().detach().numpy()
plt.hist(www)
plt.title("Hist of raw W process when coords are on a grid, var = 77183")
# %%
