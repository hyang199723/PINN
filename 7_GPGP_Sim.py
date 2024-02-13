
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
full = pd.read_csv(wk_dir + "gpgp_matern.csv")
X = np.array(full.iloc[:, 1:3])
Y = np.array(full.iloc[:, 3])
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
model_1 = RBF_train(X_train, y_train, lr=lr, epochs=1500, alpha = 0,
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
plt.title(f'Predicted value alpha = 0')
# %% Replicates
alphas = [0, 1, 100, 1000]
iters = 50
MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
for idx, alpha in enumerate(alphas):
    print(alpha)
    lr = 0.0005 # default learning rate in keras adam
    for j in range(iters):
        model_1 = RBF_train(X_train, y_train, lr=lr, epochs=1500, alpha = alpha,
                          device = device, centers=fixed_centers, dims = out_dim)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE.iloc[j, idx] = model1_mse
MSE.to_csv(wk_dir + "Output/GPGP_out.csv")

# %%