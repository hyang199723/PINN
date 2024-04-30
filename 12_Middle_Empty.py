
#%% Packages
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
#%%
from spde import *
# %% basis function
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

# %% Replicates
# Read data
dat = np.array(pd.read_csv(wk_dir + "Data/matern_02_1_1.csv", index_col=False, header = None))
original_dimension = (8000, 3, 100)
dat_full = dat.reshape(original_dimension)


lr = 0.002 # default learning rate in keras adam
nnn = 50000 # Numbr of discrete grid of points to evaluate kde
lower = -800
upper = 800
x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
theoretical_pdf = norm.pdf(x, 0, 202)

# Reserve x = [0.35, 0.65], y = [0.4, 0.6] area for testing
def mid_empty_split(X, Y):
    test_row = (X[:, 0] >= 0.35) & (X[:, 0] <= 0.65) & (X[:, 1] >= 0.4) & (X[:, 1] <= 0.6)
    train_row = ~test_row
    X_train = X[train_row, :]
    X_test = X_val = X[test_row, :]
    y_train = Y[train_row]
    y_test = y_val = Y[test_row]
    return X_train, X_val, X_test, y_train, y_val, y_test

sub = dat_full[0:8000, :, 0]
X = sub[:, 0:2]
Y = sub[:, 2]
X_train, X_val, X_test, y_train, y_val, y_test = mid_empty_split(X, Y)

plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], s = 20, c = y_train)
plt.title("Training data")
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], s = 20, c = y_test)
plt.title("Testing data")


#%% Run the model
alphas = [0, 10, 100, 1000, 100000]#
iters = 50
MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
for i in range(iters):
    print(i)
    sub = dat_full[0:4000, :, i]
    X = sub[:, 0:2]
    Y = sub[:, 2]
    X_train, X_val, X_test, y_train, y_val, y_test = mid_empty_split(X, Y)
    for idx, alpha in enumerate(alphas):
        #print(alpha)
        model_1, density, W = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=2000, alpha = alpha,
                          device = device, centers=fixed_centers, dims = out_dim, theory = theoretical_pdf)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE.iloc[i, idx] = model1_mse
MSE.to_csv(wk_dir + "Output_correct/Mid_4000sample.csv")
#d = np.exp(density)
#plt.plot(d)
#plt.title("Residual SPDE density for alpha=100000, iters = 3500, mse=0.22")

#www = W.cpu().detach().numpy()
#plt.hist(www)
# %%
