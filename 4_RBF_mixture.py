
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
# %% Replicates
dat = np.array(pd.read_csv(wk_dir + "Data/mixture_2500_02_1_1.csv", index_col=False, header = None))
original_dimension = (2500, 3, 100)
dat_full = dat.reshape(original_dimension)
# Visualize train and test

plt.scatter(dat_full[:,0,0], dat_full[:,1,0], s = 20, c = dat_full[:,2,0])
#%% Sample size = 1300
alphas = [0, 0.15, 0.3, 0.5, 1, 10]
iters = 30
MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
lr = 0.003 # default learning rate in keras adam
MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
nnn = 5000 # Numbr of discrete grid of points to evaluate kde
lower = -5
upper = 5
x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
theoretical_pdf = norm.pdf(x, 0, 1)
for i in range(iters):
    print(i)
    sub = dat_full[0:1300, :, i]
    X = sub[:, 0:2]
    Y = sub[:, 2]
    X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
    for idx, alpha in enumerate(alphas):
        #print(alpha)
        model_1, density = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=3500, alpha = alpha,
                          device = device, centers=fixed_centers, dims = out_dim, theory = theoretical_pdf)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE.iloc[i, idx] = model1_mse
MSE.to_csv(wk_dir + "Output_New/Mixture_longer_chain_1300.csv")

# %% Sample size = 500
alphas = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 10]
iters = 60
MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
lr = 0.0005 # default learning rate in keras adam
MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
for i in range(iters):
    print(i)
    sub = dat_full[0:500, :, i]
    X = sub[:, 0:2]
    Y = sub[:, 2]
    X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
    for idx, alpha in enumerate(alphas):
        #print(alpha)
        model_1 = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=1700, alpha = alpha,
                          device = device, centers=fixed_centers, dims = out_dim)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE.iloc[i, idx] = model1_mse
MSE.to_csv(wk_dir + "Output_New/Mixture_MSE_500.csv")

#%% Sample size = 2000
alphas = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 10]
iters = 60
MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
lr = 0.0005 # default learning rate in keras adam
MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
for i in range(iters):
    print(i)
    sub = dat_full[0:2000, :, i]
    X = sub[:, 0:2]
    Y = sub[:, 2]
    X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
    for idx, alpha in enumerate(alphas):
        #print(alpha)
        model_1 = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=1700, alpha = alpha,
                          device = device, centers=fixed_centers, dims = out_dim)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE.iloc[i, idx] = model1_mse
MSE.to_csv(wk_dir + "Output_New/Mixture_MSE_2000.csv")
