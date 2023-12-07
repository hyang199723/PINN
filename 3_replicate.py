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
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')
import scipy.stats as stats
import pylab
#%% Simulate 1-D stationary data:
# https://github.com/aleksada/DeepKriging/blob/master/1D_GP.ipynb
N = 1000 ##Sample Size
P = 1 ##Covariates
M = 100 ##replicates
X = np.array([np.ones(N)]).T ##Design matrix
noise_var = 0.01
rho = 0.1
spatial_var = 1
s = np.linspace(0,1,N).reshape(-1,1)
mu = np.ones(N).reshape(-1,1) # vector of the means
nugget = np.eye(N) * noise_var ##Nugget matrix

distance = distance_matrix(s.reshape(-1, 1), s.reshape(-1, 1))
corr = np.exp(-distance / rho)
cov_mat = spatial_var * corr + nugget
np.random.seed(1)
y = np.random.multivariate_normal(mu[:,0],cov_mat,M).T
#%% Visualize observations
plt.plot(s,y[:,0],".",mew=1.5)
plt.show()

#%% Generate basis function
num_basis = [10,19,37,73]
knots = [np.linspace(0,1,i) for i in num_basis]
##Wendland kernel
K = 0 ## basis size
phi = np.zeros((N, sum(num_basis)))
for res in range(len(num_basis)):
    theta = 1/num_basis[res]*2.5
    for i in range(num_basis[res]):
        d = np.absolute(s-knots[res][i])/theta
        for j in range(len(d)):
            if d[j] >= 0 and d[j] <= 1:
                phi[j,i + K] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
            else:
                phi[j,i + K] = 0
    K = K + num_basis[res]
X_full = np.concatenate((s, X, phi), axis = 1)

# Diff.
# 280(s-s0)^7 - 1344(s-s0)^6 + 2520(s-s0)^5 - 2240(s-s0)^4 + 840(s-s0)^3 - 56(s-s0)

#%% Train_test split
Xfull_train, Xfull_test, y_train, y_test = random_split(X_full, y)
s_train = Xfull_train[:, 0].reshape(-1, 1)
s_test = Xfull_test[:, 0].reshape(-1, 1)
X_train = Xfull_train[:, 1].reshape(-1, 1)
X_test = Xfull_test[:, 1].reshape(-1, 1)
phi_train = Xfull_train[:, 2:]
phi_test = Xfull_test[:, 2:]
# Visualize train and test
plt.plot(s_train, y_train[:, 0], ".", mew=1.5)
plt.title("Training data")
plt.show()
plt.plot(s_test, y_test[:, 0], ".", mew=1.5)
plt.title("Testing data")
plt.show()
# %% First, replicate paper result
# Train with X = 1 on DNN
layers = [1, 100, 100, 100, 100, 100, 100, 100, 1]
lr = 0.001 # default learning rate in keras adam
X_train_1 = X_train
model_1 = model_train_dnn(X_train_1, y_train[:, 0], layers = layers, 
                          lr=lr, epochs=300, device = device)
# Get RMSE
X_test_tc = torch.tensor(X_test).float().to(device)
y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
pylab.plot(s_test, y_test[:, 0], ".", label = "observation")
pylab.plot(s_test, y0_model1, ".", label = "DNN with intercept")
pylab.legend(loc='upper right')
pylab.show()
model1_mse = np.mean((y_test[:, 0] - y0_model1)**2)
# %% Train with s and X
layers = [2, 100, 100, 100, 100, 100, 100, 100, 1]
lr = 0.001 # default learning rate in keras adam

X_train_2 = np.hstack((X_train, s_train))
X_test_2 = np.hstack((X_test, s_test))
X_test_2_tc = torch.tensor(X_test_2).float().to(device)
model_2 = model_train_dnn(X_train_2, y_train[:, 0], layers = layers, 
                          lr=lr, epochs=300, device = device)
y0_model2 = model_2(X_test_2_tc).cpu().detach().numpy().reshape(-1)
pylab.plot(s_test, y_test[:, 0], ".", label = "observation")
pylab.plot(s_test, y0_model2, ".", label = "DNN with intercept and coordiante")
pylab.legend(loc='upper right')
pylab.show()
model2_mse = np.mean((y_test[:, 0] - y0_model2)**2)
# %% Train with X and basis functions
k = 139
layers = [k+1, 100, 100, 100, 100, 100, 100, 100, 1]
lr = 0.001
X_train_3 = np.hstack((X_train, phi_train))
X_test_3 = np.hstack((X_test, phi_test))
X_test_3_tc = torch.tensor(X_test_3).float().to(device)
model_3 = model_train_dnn(X_train_3, y_train[:, 0], layers = layers, 
                          lr=lr, epochs=300, device = device)
y0_model3 = model_3(X_test_3_tc).cpu().detach().numpy().reshape(-1)
pylab.plot(s_test, y_test[:, 0], ".", label = "observation")
pylab.plot(s_test, y0_model3, ".", label = "DeepKriging")
pylab.legend(loc='upper right')
pylab.show()
model3_mse = np.mean((y_test[:, 0] - y0_model3)**2)
# %% Train with PINN
k = 139
layers = [k+1, 100, 100, 100, 100, 100, 100, 100, 1]
lr = 0.001
X_train_4 = np.hstack((X_train, phi_train))
X_test_4 = np.hstack((X_test, phi_test))
X_test_4_tc = torch.tensor(X_test_4).float().to(device)
model_4 = model_train_pinn(X_train_4, y_train[:, 0], layers = layers, 
                          lr=lr, epochs=300, alpha = 10, momentum=0, device = device)
y0_model4 = model_4(X_test_4_tc).cpu().detach().numpy().reshape(-1)
pylab.plot(s_test, y_test[:, 0], ".", label = "observation")
pylab.plot(s_test, y0_model4, ".", label = "PINN")
pylab.legend(loc='upper right')
pylab.show()
model4_mse = np.mean((y_test[:, 0] - y0_model4)**2)

# %% Search over alpha
k = 139
layers = [k+1, 100, 100, 100, 100, 100, 100, 100, 1]
lr = 0.0005
X_train_4 = np.hstack((X_train, phi_train))
X_test_4 = np.hstack((X_test, phi_test))
X_test_4_tc = torch.tensor(X_test_4).float().to(device)
alphas = [0, 1, 10, 100, 1000]
mse = np.zeros(len(alphas))
for i, alpha in enumerate(alphas):
    model_4 = model_train_pinn(X_train_4, y_train[:, 0], layers = layers, 
                          lr=lr, epochs=300, alpha = alpha, momentum=0, device = device)
    y0_model4 = model_4(X_test_4_tc).cpu().detach().numpy().reshape(-1)
    model4_mse = np.mean((y_test[:, 0] - y0_model4)**2)
    mse[i] = model4_mse

# %%
