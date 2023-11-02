# How the weight parameter might affect the performance of PINN
# Simulate a stationary process and apply NN
#%% Packages
import sys
# wk_dir = "/Users/hongjianyang/PINN_SPDE/"
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
    

#%% Read data
data = pd.read_csv("non_stat_200.csv", index_col=None)
data = np.array(data.drop(data.columns[0], axis = 1))
X = data[:, 0:3]
Y = data[:, 3]

X_train, X_test, y_train, y_test = random_split(X, Y)

alphas = [35, 50, 60]
iters = 10 # Number of replications for each alpha
nnn = len(alphas)
MSE = np.zeros(nnn)
layers = [3, 500, 400, 300, 1]
lr = 0.01
epochs = 18000
X_test = torch.from_numpy(X_test).float().to(device)

for i in range(0, nnn):
    for j in range(0, iters):
        print(i)
        print(j)
        alpha = alphas[i]
        model = model_train_pinn(X_train, y_train, layers, lr, epochs, alpha, device)
        y_pred_tc = model(X_test)
        y_pred = y_pred_tc.cpu().detach().numpy().reshape(-1)
        mse = np.mean((y_pred - y_test)**2)
        MSE[i] += mse
#%%
temp = MSE / iters
plt.plot(alphas, temp)