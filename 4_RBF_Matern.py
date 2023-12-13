
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


#%% Simulate a 2-D stationary data
N = 1000
P = 2
noise_var = 0.1
rho = 2
nu = 1
kappa = (8 * nu)**(0.5) / rho
spatial_var = 1
X, Y = gen_matern(N, rho, spatial_var, noise_var, nu)
X = X[:, 1:3] # Only need coors
# RBF centers
num_basis = [3,5,6,8]
squared = [i**2 for i in num_basis]
out_dim = np.sum(squared)
fixed_centers = torch.randn(out_dim, 2)
sum = 0
for i in num_basis:
    loc = np.linspace(0,1,i)
    x = np.array([(x, y) for x in loc for y in loc])
    fixed_centers[sum:sum+i**2] = torch.tensor(x)
    sum += i**2

#%% PLots
X_train, X_test, y_train, y_test = random_split(X, Y)
# Visualize train and test
plt.subplot(1,2,1)
plt.scatter(X_train[:,0], X_train[:,1], s = 20, c = y_train)
plt.title("Training data")
plt.subplot(1,2,2)
plt.scatter(X_test[:,0], X_test[:,1], s = 20, c = y_test)
plt.title("Testing data")


# %%
lr = 0.0005 # default learning rate in keras adam
model_1 = RBF_train(X_train, y_train, lr=lr, epochs=1500, alpha = 10,
                          device = device, centers=fixed_centers, dims = out_dim)
#%%
# Get RMSE
X_test_tc = torch.tensor(X_test).float().to(device)
y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
plt.subplot(1,2,1)
plt.scatter(X_test[:, 0], X_test[:, 1], s = 20, c = y_test)
plt.title("Testing data")
plt.subplot(1,2,2)
plt.scatter(X_test[:, 0], X_test[:, 1], s = 20, c = y0_model1)
model1_mse = np.mean((y_test - y0_model1)**2)
plt.title(f'Predicted value; MSE = {model1_mse}')
# %% Replicates
alphas = [0, 0.5, 1, 2, 4, 8, 16, 32, 64, 100, 256, 1000]
iters = 100
MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
for idx, alpha in enumerate(alphas):
    print(alpha)
    X, Y = gen_stat(N, rho, spatial_var, noise_var)
    X = X[:, 1:3]
    X_train, X_test, y_train, y_test = random_split(X, Y)
    lr = 0.0005 # default learning rate in keras adam
    for j in range(iters):
        print(j)
        model_1 = RBF_train(X_train, y_train, lr=lr, epochs=1500, alpha = alpha,
                          device = device, centers=fixed_centers, dims = out_dim)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE.iloc[j, idx] = model1_mse


# %%
