
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

#%% Basis function
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
#Ns = [200 * 2**i for i in range(0, 5)]
Ns = [400, 600, 800, 1000, 1500]
iters = 30
epochs = 1600
#%% Replicate, alpha = 0
dat = np.array(pd.read_csv(wk_dir + "Data/matern_02_1_5.csv", index_col=False, header = None))
MSE_0 = pd.DataFrame(data = 0.0, index = range(iters), columns = Ns)
lr = 0.0005 # default learning rate in keras adam
for idx, n in enumerate(Ns):
    print(n)
    size = n * 3
    subdat = dat[0:size, :]
    original_dimension = (n, 3, 100)
    dat_full = subdat.reshape(original_dimension)
    for j in range(iters):
        sub = dat_full[:, :, j]
        X = sub[:, 0:2]
        Y = sub[:, 2]
        X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
        model_1 = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=epochs, alpha = 0,
                          device = device, centers=fixed_centers, dims = out_dim)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE_0.iloc[j, idx] = model1_mse
MSE_0.to_csv(wk_dir + "Output_New/Sample_alpha0.csv")
#%% Alpha = 0.03, 0.08, 0.2, 1
dat = np.array(pd.read_csv(wk_dir + "Data/matern_02_1_5.csv", index_col=False, header = None))
MSE_1 = pd.DataFrame(data = 0.0, index = range(iters), columns = Ns)
lr = 0.0005 # default learning rate in keras adam
for idx, n in enumerate(Ns):
    print(n)
    size = n * 3
    subdat = dat[0:size, :]
    original_dimension = (n, 3, 100)
    dat_full = subdat.reshape(original_dimension)
    for j in range(iters):
        sub = dat_full[:, :, j]
        X = sub[:, 0:2]
        Y = sub[:, 2]
        X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
        model_1 = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=epochs, alpha = 0.03,
                          device = device, centers=fixed_centers, dims = out_dim)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE_1.iloc[j, idx] = model1_mse
MSE_1.to_csv(wk_dir + "Output_New/Sample_alpha003.csv")

#%% Alpha = 0.08, 0.2, 1
dat = np.array(pd.read_csv(wk_dir + "Data/matern_02_1_5.csv", index_col=False, header = None))
MSE_2 = pd.DataFrame(data = 0.0, index = range(iters), columns = Ns)
lr = 0.0005 # default learning rate in keras adam
for idx, n in enumerate(Ns):
    print(n)
    size = n * 3
    subdat = dat[0:size, :]
    original_dimension = (n, 3, 100)
    dat_full = subdat.reshape(original_dimension)
    for j in range(iters):
        sub = dat_full[:, :, j]
        X = sub[:, 0:2]
        Y = sub[:, 2]
        X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
        model_1 = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=epochs, alpha = 0.08,
                          device = device, centers=fixed_centers, dims = out_dim)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE_2.iloc[j, idx] = model1_mse
MSE_2.to_csv(wk_dir + "Output_New/Sample_alpha008.csv")
        
#%% Alpha = 0.2, 1
dat = np.array(pd.read_csv(wk_dir + "Data/matern_02_1_5.csv", index_col=False, header = None))
MSE_3 = pd.DataFrame(data = 0.0, index = range(iters), columns = Ns)
lr = 0.0005 # default learning rate in keras adam
for idx, n in enumerate(Ns):
    print(n)
    size = n * 3
    subdat = dat[0:size, :]
    original_dimension = (n, 3, 100)
    dat_full = subdat.reshape(original_dimension)
    for j in range(iters):
        sub = dat_full[:, :, j]
        X = sub[:, 0:2]
        Y = sub[:, 2]
        X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
        model_1 = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=epochs, alpha = 0.2,
                          device = device, centers=fixed_centers, dims = out_dim)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE_3.iloc[j, idx] = model1_mse
MSE_3.to_csv(wk_dir + "Output_New/Sample_alpha02.csv")

#%% Alpha = 1
dat = np.array(pd.read_csv(wk_dir + "Data/matern_02_1_5.csv", index_col=False, header = None))
MSE_4 = pd.DataFrame(data = 0.0, index = range(iters), columns = Ns)
lr = 0.0005 # default learning rate in keras adam
for idx, n in enumerate(Ns):
    print(n)
    size = n * 3
    subdat = dat[0:size, :]
    original_dimension = (n, 3, 100)
    dat_full = subdat.reshape(original_dimension)
    for j in range(iters):
        sub = dat_full[:, :, j]
        X = sub[:, 0:2]
        Y = sub[:, 2]
        X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
        model_1 = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=epochs, alpha = 1,
                          device = device, centers=fixed_centers, dims = out_dim)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE_4.iloc[j, idx] = model1_mse
MSE_4.to_csv(wk_dir + "Output_New/Sample_alpha1.csv")
