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
rho = 3
vvv = 2
nugget = 1
mmtm = 0
# Sample size
Ns = [1000 * i for i in range(1,6)]
# alphas
alphas = [0, 10, 100, 1000]
iters = 1
MSE = pd.DataFrame(data = 0.0, index = Ns, columns = alphas)
for iii in range(5):
    N = Ns[iii]
    for jjj in range(4):
        alpha = alphas[jjj]
        X, Y = gen_non_same(N, rho, vvv, nugget)
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
        layers = [dims, 50, 50, 50, 1]
        lr = 0.002
        epochs = int(N / 5)
        mse_total = 0
        for j in range(iters):
            model = model_train_pinn(X_train, y_train, layers, lr, epochs, alpha = 0, momentum = mmtm, device = device)
            y_pred_tc = model(X_test_tc)
            y_pred = y_pred_tc.cpu().detach().numpy().reshape(-1)
            mse = np.mean((y_pred - y_test)**2)
            mse_total += mse
        mse_total /= iters
        MSE.iloc[iii,jjj] = mse_total



#%% PINN

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
