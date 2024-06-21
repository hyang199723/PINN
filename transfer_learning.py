#%% Packages
import sys
wk_dir = '/Users/hongjianyang/PINN/'
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
from spde import *
iters = 100
# %%
n = 55#, 8, 10, 15]
N = int(n**2) ## sample size
M = 1 ## Number of replicate
coord1 = np.linspace(0,1,n)
coord2 = np.linspace(0,1,n)
P = 1
X = np.array([np.ones(N)]).T
s1,s2 = np.meshgrid(coord1,coord2)
s = np.vstack((s1.flatten(),s2.flatten())).T
noise = np.random.normal(0, 0.001, (N, 2))
coords = s
s = s + noise
y = np.sin(30*((s[:,0]+s[:,1])/2-0.9)**4)*np.cos(2*((s[:,0]+s[:,1])/2-0.9))+((s[:,0]+s[:,1])/2-0.9)/2

##Visualization
y_mat = y.reshape(n,n)
fig, ax = plt.subplots()
im = ax.imshow(y_mat , extent=[0, 1, 0, 1], origin="lower",
               vmax=y_mat .max(), vmin=y_mat .min())
plt.xlabel('$s_1$')
plt.ylabel('$s_2$')
plt.title('Sample Non-stationary Process')
plt.colorbar(im)
plt.show()
# %% Train the base model on non stationary data
alpha = 0
#alphas = [0]# 0, 10, 100, 1000, 
# Number of layers and neurons
#layers = 3
neurons = 100
layer = 5
lr = 0.002 # default learning rate in keras adam
nnn = 5000 # Numbr of discrete grid of points to evaluate kde
lower = -800
upper = 800
KL_params = [nnn, lower, upper]
x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
theoretical_pdf = norm.pdf(x, 0, 202)
rho = 0.2
num_centers = [9,25,36,64]
#num_centers = [20**2,38**2,70**2]
eee = 1500 # NN iterations
X = s
Y = y

model_1, _, _ = RBF_train(X, X, Y, Y, lr=lr, epochs=eee, alpha = 0,
                          device = device, n_centers=num_centers, 
                          theory = theoretical_pdf, rho = rho, kl_params = KL_params,
                          layers = layer, neurons=neurons)

# %% Pre-train on matern data with 3000 samples
alpha = 0# 0, 10, 100, 1000, 
lr = 0.002 # default learning rate in keras adam
nnn = 5000 # Numbr of discrete grid of points to evaluate kde
lower = -800
upper = 800
KL_params = [nnn, lower, upper]
num_centers = [9,25,36,64]
x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
theoretical_pdf = norm.pdf(x, 0, 202)
rho = 0.2
#num_centers = [9,25,36,64]
# Number of layers and neurons
neurons = 100
layer = 5
eee = 1500
sub = dat_full[0:3000, :, 0]
X = sub[:, 0:2]
Y = sub[:, 2]
model_2_3000, _, _ = RBF_train(X, X, Y, Y, lr=lr, epochs=eee, alpha = 0,
                          device = device, n_centers=num_centers, 
                          theory = theoretical_pdf, rho = rho, kl_params = KL_params,
                          layers = layer, neurons=neurons)

# %% Pre-train on matern data with 6000 samples
sub = dat_full[0:6000, :, 0]
X = sub[:, 0:2]
Y = sub[:, 2]
model_2_6000, _, _ = RBF_train(X, X, Y, Y, lr=lr, epochs=eee, alpha = 0,
                          device = device, n_centers=num_centers, 
                          theory = theoretical_pdf, rho = rho, kl_params = KL_params,
                          layers = layer, neurons=neurons)

# %% Pre-train on matern 2 with 3000 samples
dat = np.array(pd.read_csv(wk_dir + "Data/matern_02_1_5.csv", index_col=False, header = None))
original_dimension = (5000, 3, 100)
dat_full = dat.reshape(original_dimension)

alpha = 0# 0, 10, 100, 1000, 
lr = 0.002 # default learning rate in keras adam
nnn = 5000 # Numbr of discrete grid of points to evaluate kde
lower = -800
upper = 800
KL_params = [nnn, lower, upper]
num_centers = [9,25,36,64]
x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
theoretical_pdf = norm.pdf(x, 0, 202)
rho = 0.2
#num_centers = [9,25,36,64]
# Number of layers and neurons
neurons = 100
layer = 5
eee = 1500
sub = dat_full[0:3000, :, 0]
X = sub[:, 0:2]
Y = sub[:, 2]
model_3_3000, _, _ = RBF_train(X, X, Y, Y, lr=lr, epochs=eee, alpha = 0,
                          device = device, n_centers=num_centers, 
                          theory = theoretical_pdf, rho = rho, kl_params = KL_params,
                          layers = layer, neurons=neurons)
#%% 
##################
##################
##################
stat_3000_foundation = copy.deepcopy(model_2_3000)
stat_6000_foundation = copy.deepcopy(model_2_6000)
case2_3000 = copy.deepcopy(model_3_3000)


dat = np.array(pd.read_csv(wk_dir + "Data/matern_02_1_1.csv", index_col=False, header = None))
original_dimension = (8000, 3, 100)
dat_full = dat.reshape(original_dimension)



# %% Case 4: Transfer learning result on smaller data sizes
import copy
nonstat_foundation = copy.deepcopy(model_1)

sizes = [0]
MSE_nonstat = pd.DataFrame(data = 0.0, index = range(iters), columns = sizes)
eee = 1500
for i in range(iters):
    for (idx, n) in enumerate(sizes):
        n = 18
        N = int(n**2) ## sample size
        M = 1 ## Number of replicate
        coord1 = np.linspace(0,1,n)
        coord2 = np.linspace(0,1,n)
        P = 1
        X = np.array([np.ones(N)]).T
        s1,s2 = np.meshgrid(coord1,coord2)
        s = np.vstack((s1.flatten(),s2.flatten())).T
        noise = np.random.normal(0, 0.001, (N, 2))
        s = s + noise
        y = np.sin(30*((s[:,0]+s[:,1])/2-0.9)**4)*np.cos(2*((s[:,0]+s[:,1])/2-0.9))+((s[:,0]+s[:,1])/2-0.9)/2
        X = s
        Y = y
        X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
        nonstat_model = copy.deepcopy(nonstat_foundation)
        model, _, _ = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=eee, alpha = 0,
                          device = device, n_centers=num_centers, 
                          theory = theoretical_pdf, rho = rho, kl_params = KL_params,
                          layers = layer, neurons=neurons, raw_model = nonstat_model)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE_nonstat.iloc[i, idx] = model1_mse
# %% Case 1: Transfer 3000 base model on 300 target data
sizes = [0]
MSE_stat = pd.DataFrame(data = 0.0, index = range(iters), columns = sizes)
eee = 1500
for i in range(iters):
    print(i)
    for (idx, n) in enumerate(sizes):
        sub = dat_full[0:300, :, i]
        X = sub[:, 0:2]
        Y = sub[:, 2]
        X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
        stat_model = copy.deepcopy(stat_3000_foundation)
        model, _, _ = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=eee, alpha = 0,
                          device = device, n_centers=num_centers, 
                          theory = theoretical_pdf, rho = rho, kl_params = KL_params,
                          layers = layer, neurons=neurons, raw_model = stat_model)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE_stat.iloc[i, idx] = model1_mse
        
# %% Case 2: Transfer 3000 base model on 300 target data
sizes = [0]
MSE_stat = pd.DataFrame(data = 0.0, index = range(iters), columns = sizes)
eee = 1500
for i in range(iters):
    print(i)
    for (idx, n) in enumerate(sizes):
        sub = dat_full[0:300, :, i]
        X = sub[:, 0:2]
        Y = sub[:, 2]
        X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
        stat_model = copy.deepcopy(case2_3000)
        model, _, _ = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=eee, alpha = 0,
                          device = device, n_centers=num_centers, 
                          theory = theoretical_pdf, rho = rho, kl_params = KL_params,
                          layers = layer, neurons=neurons, raw_model = stat_model)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE_stat.iloc[i, idx] = model1_mse
        
        
# %% Case 3: Transfer 6000 base model on 300 target data
sizes = [0]
MSE_stat = pd.DataFrame(data = 0.0, index = range(iters), columns = sizes)
eee = 1500
for i in range(iters):
    print(i)
    for (idx, n) in enumerate(sizes):
        n = n**2
        sub = dat_full[0:300, :, i]
        X = sub[:, 0:2]
        Y = sub[:, 2]
        X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
        stat_model = copy.deepcopy(stat_6000_foundation)
        model, _, _ = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=eee, alpha = alpha,
                          device = device, n_centers=num_centers, 
                          theory = theoretical_pdf, rho = rho, kl_params = KL_params,
                          layers = layer, neurons=neurons, raw_model = stat_model)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE_stat.iloc[i, idx] = model1_mse


#%% Target only
sizes = [0]
MSE_nonstat = pd.DataFrame(data = 0.0, index = range(iters), columns = sizes)
eee = 1500
for i in range(iters):
    print(i)
    for (idx, n) in enumerate(sizes):
        n = 18
        N = int(n**2) ## sample size
        M = 1 ## Number of replicate
        coord1 = np.linspace(0,1,n)
        coord2 = np.linspace(0,1,n)
        P = 1
        X = np.array([np.ones(N)]).T
        s1,s2 = np.meshgrid(coord1,coord2)
        s = np.vstack((s1.flatten(),s2.flatten())).T
        noise = np.random.normal(0, 0.001, (N, 2))
        s = s + noise
        y = np.sin(30*((s[:,0]+s[:,1])/2-0.9)**4)*np.cos(2*((s[:,0]+s[:,1])/2-0.9))+((s[:,0]+s[:,1])/2-0.9)/2
        X = s
        Y = y
        X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
        model, _, _ = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=eee, alpha = 0,
                          device = device, n_centers=num_centers, 
                          theory = theoretical_pdf, rho = rho, kl_params = KL_params,
                          layers = layer, neurons=neurons)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE_nonstat.iloc[i, idx] = model1_mse
