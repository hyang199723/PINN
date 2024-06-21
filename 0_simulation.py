# Simulate and save data of different spatial covariance structure

#%% Packages
import sys
# wk_dir = "/Users/hongjianyang/PINN/"
wk_dir = "/share/bjreich/hyang23/PINN"
sys.path.append(wk_dir)
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from scipy.spatial import distance_matrix
from scipy.special import gamma, kv
import scipy.stats as stats
import pylab

def Matern_Cor(nu, rho, distance):
    kappa = (8 * nu)**(0.5) / rho
    const = 1 / (2**(nu - 1) * gamma(nu))
    kd = kappa * distance
    first_term = kd**nu
    second_term = kv(nu, kd)
    second_term[np.diag_indices_from(second_term)] = 0.
    out = const * first_term * second_term
    out[np.diag_indices_from(out)] = 1.0
    return out

def gen_matern(N, rho, spatial_var, noise_var, nu):
    n = N
    length = 1
    coords = np.random.uniform(0, length, (N, 2))
    X = np.zeros((n, 3))
    X[:, 0] = 1
    X[:, 1:3] = coords

    # Exponential Correlation
    distance = distance_matrix(coords.reshape(-1, 2), coords.reshape(-1, 2))
    corr = Matern_Cor(nu, rho, distance)
    # Cholesky decomposition and generate correlated data
    L = np.linalg.cholesky(spatial_var*corr)
    z = np.random.normal(0, 1, n)
    Y = np.dot(L, z) + np.random.normal(0, noise_var, n)
    return X, Y

# %% 2-D stationary process
N = 8000
P = 2
noise_std = 0.1
rho = 0.2
nu = 1
kappa = (8 * nu)**(0.5) / rho
spatial_var = 1

iters = 100
matern_data = np.zeros((N, 3, iters))
for i in range(iters):
    print(i)
    X, Y = gen_matern(N, rho, spatial_var, noise_std, nu)
    X = X[:, 1:3] # Only need coors
    matern_data[:, 0:2, i] = X
    matern_data[:, 2, i] = Y
# Save to data file for comparison
matern_data_flat = matern_data.reshape(-1, matern_data.shape[-1])
np.savetxt("Data/matern_02_1_1.csv", matern_data_flat, delimiter=",")

# %% Simulate 2-D stationay process with smaller range
N = 8000
P = 2
noise_std = 0.1
rho = 0.1
nu = 1
kappa = (8 * nu)**(0.5) / rho
spatial_var = 1

iters = 100
matern_data = np.zeros((N, 3, iters))
for i in range(iters):
    print(i)
    X, Y = gen_matern(N, rho, spatial_var, noise_std, nu)
    X = X[:, 1:3] # Only need coors
    matern_data[:, 0:2, i] = X
    matern_data[:, 2, i] = Y
# Save to data file for comparison
matern_data_flat = matern_data.reshape(-1, matern_data.shape[-1])
np.savetxt(wk_dir + "Data/matern_01_1_1.csv", matern_data_flat, delimiter=",")

# %% 2-D stationary process with higher spatial and nugget variance
N = 5000
P = 2
noise_std = 5**0.5
rho = 0.2
nu = 1
kappa = (8 * nu)**(0.5) / rho
spatial_var = 5

iters = 100
matern_data = np.zeros((N, 3, iters))
for i in range(iters):
    X, Y = gen_matern(N, rho, spatial_var, noise_std, nu)
    X = X[:, 1:3] # Only need coors
    matern_data[:, 0:2, i] = X
    matern_data[:, 2, i] = Y
# Save to data file for comparison
matern_data_flat = matern_data.reshape(-1, matern_data.shape[-1])
np.savetxt(wk_dir + "Data/matern_02_1_5.csv", matern_data_flat, delimiter=",")
# %% Generate non-stationary mixture process
N = 2500
P = 2
noise_var = 0.1
rho = 0.2
nu = 1
kappa = (8 * nu)**(0.5) / rho
spatial_var = 1

iters = 100
non_stat = np.zeros((N, 3, iters))
for i in range(iters):
    X, Y = gen_mixture(N, spatial_var, noise_var, nu1 = 3.5, nu2 = 0.5,
                   rho1 = 0.3, rho2 = 0.1)
    X = X[:, 1:3] # Only need coors
    non_stat[:, 0:2, i] = X
    non_stat[:, 2, i] = Y
non_stat_flat = non_stat.reshape(-1, non_stat.shape[-1])
np.savetxt("Data/mixture_2500_02_1_1.csv", non_stat_flat, delimiter=",")
# %% https://arxiv.org/pdf/2006.15640.pdf Spatial processes

# Proecss 2
# Y (s) = Z(s)^3 + E(s);
N = 2000
P = 2
noise_std = 0.1
rho = 0.2
nu = 1
kappa = (8 * nu)**(0.5) / rho
spatial_var = 1

iters = 100
matern_data = np.zeros((N, 3, iters))
for i in range(iters):
    print(i)
    length = 1
    coords = np.random.uniform(0, length, (N, 2))
    X = np.zeros((N, 3))
    X[:, 0] = 1
    X[:, 1:3] = coords

    # Exponential Correlation
    distance = distance_matrix(coords.reshape(-1, 2), coords.reshape(-1, 2))
    corr = Matern_Cor(nu, rho, distance)
    # Cholesky decomposition and generate correlated data
    L = np.linalg.cholesky(spatial_var*corr)
    z = np.random.normal(0, 1, N)
    # Process 3
    Y = np.dot(L, z)**3 + np.random.normal(0, noise_std, N)

    X = X[:, 1:3] # Only need coors
    matern_data[:, 0:2, i] = X
    matern_data[:, 2, i] = Y
# Save to data file for comparison
matern_data_flat = matern_data.reshape(-1, matern_data.shape[-1])
np.savetxt("Data/process_2.csv", matern_data_flat, delimiter=",")

# %% Process 3
# Y(s) = q[Φ{Z(s)/ 3}] + E(s)
N = 2000
P = 2
noise_std = 0.1
rho = 0.2
nu = 1
kappa = (8 * nu)**(0.5) / rho
spatial_var = 1

iters = 100
matern_data = np.zeros((N, 3, iters))
for i in range(iters):
    print(i)
    length = 1
    coords = np.random.uniform(0, length, (N, 2))
    X = np.zeros((N, 3))
    X[:, 0] = 1
    X[:, 1:3] = coords

    # Exponential Correlation
    distance = distance_matrix(coords.reshape(-1, 2), coords.reshape(-1, 2))
    corr = Matern_Cor(nu, rho, distance)
    # Cholesky decomposition and generate correlated data
    L = np.linalg.cholesky(spatial_var*corr)
    z = np.random.normal(0, 1, N)
    # Process 3
    zzz = np.dot(L, z)
    
    temp = stats.norm.cdf(zzz / (3**0.5))
    # Based on Python's definition, gamma(shape, scale)
    scale = 1 / (3**0.5)
    Y = stats.gamma.ppf(temp, 1, loc=0, scale = scale) + np.random.normal(0, noise_std, N)
    X = X[:, 1:3] # Only need coors
    matern_data[:, 0:2, i] = X
    matern_data[:, 2, i] = Y
# Save to data file for comparison
matern_data_flat = matern_data.reshape(-1, matern_data.shape[-1])
np.savetxt("Data/process_3.csv", matern_data_flat, delimiter=",")

# %% Process 4
# Y(s) = \sqrt(3) * Z(s) |E(s)|
N = 2000
noise_std = 0.1
rho = 0.2
nu = 1
kappa = (8 * nu)**(0.5) / rho
spatial_var = 1

iters = 100
matern_data = np.zeros((N, 3, iters))
for i in range(iters):
    print(i)
    length = 1
    coords = np.random.uniform(0, length, (N, 2))
    X = np.zeros((N, 3))
    X[:, 0] = 1
    X[:, 1:3] = coords

    # Exponential Correlation
    distance = distance_matrix(coords.reshape(-1, 2), coords.reshape(-1, 2))
    corr = Matern_Cor(nu, rho, distance)
    # Cholesky decomposition and generate correlated data
    L = np.linalg.cholesky(spatial_var*corr)
    z = np.random.normal(0, 1, N)
    # Process 4
    zzz = np.multiply(np.sqrt(3), np.dot(L, z))
    residual = np.abs(np.random.normal(0, noise_std, N))
    Y = np.multiply(zzz, residual)

    X = X[:, 1:3] # Only need coors
    matern_data[:, 0:2, i] = X
    matern_data[:, 2, i] = Y
# Save to data file for comparison
matern_data_flat = matern_data.reshape(-1, matern_data.shape[-1])
np.savetxt("Data/process_4.csv", matern_data_flat, delimiter=",")

# %% Skip Process 5 for now; Go to Process 6
# Y(s) = sqrt(ω(s)/3)  Z(s) + sqrt(1 − ω(s)) E(s) where ω(s) = Φ(sx−0.5);

N = 2000
noise_std = 0.1
rho = 0.2
nu = 1
kappa = (8 * nu)**(0.5) / rho
spatial_var = 1

iters = 100
matern_data = np.zeros((N, 3, iters))
for i in range(iters):
    print(i)
    length = 1
    coords = np.random.uniform(0, length, (N, 2))
    X = np.zeros((N, 3))
    X[:, 0] = 1
    X[:, 1:3] = coords

    # Exponential Correlation
    distance = distance_matrix(coords.reshape(-1, 2), coords.reshape(-1, 2))
    corr = Matern_Cor(nu, rho, distance)
    # Cholesky decomposition and generate correlated data
    L = np.linalg.cholesky(spatial_var*corr)
    z = np.random.normal(0, 1, N)
    # Process 6
    zzz = np.dot(L, z)
    residual = np.random.normal(0, noise_std, N)
    w = stats.norm.cdf(X[:, 1], 0.5, 0.1)
    Y = np.sqrt(w/np.sqrt(3)) * zzz + np.sqrt(1-w) * residual

    X = X[:, 1:3] # Only need coors
    matern_data[:, 0:2, i] = X
    matern_data[:, 2, i] = Y
# Save to data file for comparison
matern_data_flat = matern_data.reshape(-1, matern_data.shape[-1])
np.savetxt("Data/process_6.csv", matern_data_flat, delimiter=",")
# %%
