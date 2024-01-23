
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

# Generate uneven data set
# The data still follows a matern process
def gen_uneven(N, rho, spatial_var, noise_var, nu):
    random.seed(123)
    coords1 = np.random.beta(1, 3, size = (math.floor(N/2), 2))
    coords2 = np.random.beta(3, 1, size = (math.ceil(N/2), 2))
    coords = np.concatenate((coords1, coords2))
    X = np.zeros((N, 3))
    X[:, 0] = 1
    X[:, 1:3] = coords

    # Exponential Correlation
    distance = distance_matrix(coords.reshape(-1, 2), coords.reshape(-1, 2))
    corr = Matern_Cor(nu, rho, distance)
    # Cholesky decomposition and generate correlated data
    L = np.linalg.cholesky(spatial_var*corr)
    z = np.random.normal(0, 1, N)
    Y = np.dot(L, z) + np.random.normal(0, noise_var, N)
    return X, Y

#%% Simulate a 2-D Matern
N = 1000
P = 2
noise_var = 0.1
rho = 0.2
nu = 1
kappa = (8 * nu)**(0.5) / rho
spatial_var = 1
X, Y = gen_uneven(N, rho, spatial_var, noise_var, nu)
X = X[:, 1:3] # Only need coors

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
#%% Plot the points distribution
p = plt.scatter(X[:, 0], X[:, 1], c = Y)
plt.colorbar(p)
plt.show()
# %% Replicates
alphas = [0, 1, 100, 500, 1000]
iters = 30
MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
MSE_fixed = [0.0] * iters
MSE_kriging = [0.0] * iters
in_dims = 2
out_dims = 150
for idx, alpha in enumerate(alphas):
    print(alpha)
    X, Y = gen_uneven(N, rho, spatial_var, noise_var, nu)
    X = X[:, 1:3]
    X_train, X_test, y_train, y_test = random_split(X, Y)
    lr = 0.0005 # default learning rate in keras adam
    for j in range(iters):
        model_1 = RBFRandmTrain(X_train, y_train, lr=lr, epochs=1500, alpha = alpha,
                          device = device, in_dims = in_dims, out_dims = out_dims)
        if alpha == 0:
            model2 = RBF_train(X_train, y_train, lr=lr, epochs=1500, alpha = alpha,
                          device = device, centers=fixed_centers, dims = out_dim)
            X2_test_tc = torch.tensor(X_test).float().to(device)
            y0_model2 = model2(X2_test_tc).cpu().detach().numpy().reshape(-1)
            model2_mse = np.mean((y_test - y0_model2)**2)
            MSE_fixed[j] = model2_mse
            # Kriging
            kriging_hat = Kriging(X_train, X_test, y_train, 0.2)
            MSE_kriging[j] = np.mean((kriging_hat - y_test)**2)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE.iloc[j, idx] = model1_mse
MSE.to_csv(wk_dir + "random_center_uneven_MSE.csv")
MSE_fixed = pd.DataFrame(MSE_fixed)
MSE_fixed.to_csv(wk_dir + "random_center_uneven_compare_MSE.csv")
# %%
