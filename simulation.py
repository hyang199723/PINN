# Simulate a stationary process and apply NN
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
    

#%% Generate data
N = 5000
rho = 3
vvv = 2
nugget = 1
X, Y = gen_non_same(N, rho, vvv, nugget)
plt.scatter(X[:,1], X[:,2], s = 20, c = Y)
plt.colorbar()
# Create basis function
s = np.linspace(0,1,N).reshape(-1,1)
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
# RBF
X = np.hstack((X, phi[:, 1:140]))
dims = X.shape[1]
X_train, X_test, y_train, y_test = random_split(X, Y)
X_test_tc = torch.from_numpy(X_test).float().to(device)
#%% PINN
layers = [dims, 50, 50, 50, 1]
lr = 0.002
mmtm = 0
epochs = 40
model = model_train_pinn(X_train, y_train, layers, lr, epochs, alpha = 0, momentum = mmtm, device = device)
y_pred_tc = model(X_test_tc)
y_pred = y_pred_tc.cpu().detach().numpy().reshape(-1)
mse = np.mean((y_pred - y_test)**2)
v = np.var(y_test)
print(f'The testing MSE is {mse}')
print(f'Testing data varianec is {v}')
#%%
# Plot testing data
plt.subplot(1,2,1)
plt.scatter(X_test[:,1], X_test[:,2], s = 20, c = y_pred)
plt.title("Predicted value")
plt.subplot(1,2,2)
plt.scatter(X_test[:,1], X_test[:,2], s = 20, c = y_test)
plt.colorbar()
plt.title("True value")
plt.show()
#%%
# Plot training data
X_train_tc = torch.from_numpy(X_train).float().to(device)
y_train_tc = model(X_train_tc)
y_train_pred = y_train_tc.cpu().detach().numpy().reshape(-1)
train_mse = np.mean((y_train_pred - y_train)**2)
print(f'Training mse is {train_mse}')
plt.subplot(1,2,1)
plt.scatter(X_train[:,1], X_train[:,2], s = 20, c = y_train_pred)
plt.title("Predicted value of training data")
plt.subplot(1,2,2)
plt.scatter(X_train[:,1], X_train[:,2], s = 20, c = y_train)
plt.colorbar()
plt.title("True value")
plt.show()
#%%
abs_diff = np.abs(y_pred - y_test)
plt.scatter(X_test[:,1], X_test[:,2], s = 20, c = abs_diff)
plt.colorbar()
tit = "Absolute diff, mse = " + str(mse)
plt.title(tit)
#%%
# Kriging
y_pred = Kriging(X_train, X_test, y_train, N, vvv, rho)
mse = np.mean((y_pred - y_test)**2)

plt.subplot(1,2,1)
plt.scatter(X_test[:,1], X_test[:,2], s = 20, c = y_pred)
plt.title("Predicted value")
plt.subplot(1,2,2)
plt.scatter(X_test[:,1], X_test[:,2], s = 20, c = y_test)
plt.colorbar()
plt.title("True value")
plt.show()

abs_diff = np.abs(y_pred - y_test)
plt.scatter(X_test[:,1], X_test[:,2], s = 20, c = abs_diff)
plt.colorbar()
tit = "Absolute diff, mse = " + str(mse)
plt.title(tit)
#%%
# NN
model2 = model_train_dnn(X_train, y_train, layers, lr, epochs, split = "random")
y_pred_tc = model2(X_test)
y_pred = y_pred_tc.detach().numpy().reshape(-1)
plt.subplot(1,2,1)
plt.scatter(X_test[:,1], X_test[:,2], s = 20, c = y_pred)
plt.title("Predicted value")
plt.subplot(1,2,2)
plt.scatter(X_test[:,1], X_test[:,2], s = 20, c = y_test)
plt.colorbar()
plt.title("True value")
plt.show()
mse = np.mean((y_pred - y_test)**2)
abs_diff = np.abs(y_pred - y_test)
plt.scatter(X_test[:,1], X_test[:,2], s = 20, c = abs_diff)
plt.colorbar()
tit = "Absolute diff, mse = " + str(mse)
plt.title(tit)

#%% How testing MSE changes over sample size?
Ns = [2**i*10 for i in range(3,8)]
xx = len(Ns)
iters = 10
PINN_mse = [0] * xx
NN_mse = [0] * xx
Krig_mse = [0] * xx
rho = 3
vvv = 2
layers = [3, 100, 100, 100, 1]
lr = 0.01
epochs = 30000
for i in range(xx):
    # Parameters and data
    n = Ns[i]
    X, Y = gen_non_same(n, rho, vvv)
    X_train, X_test, y_train, y_test = random_split(X, Y)
    
    # PINN
    model = model_train_pinn(X_train, y_train, layers, lr, epochs)
    X_test = torch.from_numpy(X_test).to(torch.float32)
    y_pred_tc = model(X_test)
    y_pred = y_pred_tc.detach().numpy().reshape(-1)
    mse = np.mean((y_pred - y_test)**2)
    PINN_mse[i] = mse
    
    # NN
    model2 = model_train_dnn(X_train, y_train, layers, lr, epochs, split = "random")
    y_pred_tc = model2(X_test)
    y_pred = y_pred_tc.detach().numpy().reshape(-1)
    mse = np.mean((y_pred - y_test)**2)
    NN_mse[i] = mse
    
    # Kriging
    y_pred = Kriging(X_train, X_test, y_train, n, vvv, rho)
    mse = np.mean((y_pred - y_test)**2)
    Krig_mse[i] = mse





#%%
plt.plot(Ns, Krig_mse, label = "Kriging") 
plt.plot(Ns, NN_mse, label = "NN") 
plt.plot(Ns, PINN_mse, label = "PINN") 
plt.legend() 
plt.show()







