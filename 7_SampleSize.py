
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
rho = 0.2
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

#%% Split
X_train, X_test, y_train, y_test = random_split(X, Y)
# %% Replicates
Ns = [200 * 2**i for i in range(0, 7)]
iters = 100
MSE_0 = pd.DataFrame(data = 0.0, index = range(iters), columns = Ns)
for idx, n in enumerate(Ns):
    print(n)
    X, Y = gen_stat(n, rho, spatial_var, noise_var)
    X = X[:, 1:3]
    X_train, X_test, y_train, y_test = random_split(X, Y)
    lr = 0.0005 # default learning rate in keras adam
    for j in range(iters):
        model_1 = RBF_train(X_train, y_train, lr=lr, epochs=1500, alpha = 0,
                          device = device, centers=fixed_centers, dims = out_dim)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE_0.iloc[j, idx] = model1_mse
MSE_0.to_csv(wk_dir + "Sample_alpha0.csv")

MSE_1 = pd.DataFrame(data = 0.0, index = range(iters), columns = Ns)
for idx, n in enumerate(Ns):
    print(n)
    X, Y = gen_stat(n, rho, spatial_var, noise_var)
    X = X[:, 1:3]
    X_train, X_test, y_train, y_test = random_split(X, Y)
    lr = 0.0005 # default learning rate in keras adam
    for j in range(iters):
        model_1 = RBF_train(X_train, y_train, lr=lr, epochs=1500, alpha = 1,
                          device = device, centers=fixed_centers, dims = out_dim)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE_1.iloc[j, idx] = model1_mse
MSE_1.to_csv(wk_dir + "Sample_alpha1.csv")


MSE_100 = pd.DataFrame(data = 0.0, index = range(iters), columns = Ns)
for idx, n in enumerate(Ns):
    print(n)
    X, Y = gen_stat(n, rho, spatial_var, noise_var)
    X = X[:, 1:3]
    X_train, X_test, y_train, y_test = random_split(X, Y)
    lr = 0.0005 # default learning rate in keras adam
    for j in range(iters):
        model_1 = RBF_train(X_train, y_train, lr=lr, epochs=1500, alpha = 100,
                          device = device, centers=fixed_centers, dims = out_dim)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE_100.iloc[j, idx] = model1_mse
MSE_100.to_csv(wk_dir + "Sample_alpha100.csv")

MSE_1000 = pd.DataFrame(data = 0.0, index = range(iters), columns = Ns)
for idx, n in enumerate(Ns):
    print(n)
    X, Y = gen_stat(n, rho, spatial_var, noise_var)
    X = X[:, 1:3]
    X_train, X_test, y_train, y_test = random_split(X, Y)
    lr = 0.0005 # default learning rate in keras adam
    for j in range(iters):
        model_1 = RBF_train(X_train, y_train, lr=lr, epochs=1500, alpha = 1000,
                          device = device, centers=fixed_centers, dims = out_dim)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE_1000.iloc[j, idx] = model1_mse
MSE_1000.to_csv(wk_dir + "Sample_alpha1000.csv")


