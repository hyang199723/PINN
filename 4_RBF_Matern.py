
#%% Packages
import sys
#wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/PINN/
# wk_dir = 'C://Users//hyang23//PINN//'
#wk_dir = '/Users/hongjianyang/PINN/'
wk_dir = "/share/bjreich/hyang23/PINN/"
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
    device = torch.device('cuda:2')
else:
    device = torch.device('cpu')
import scipy.stats as stats
from spde import *
import time
print(device)
# %% Replicates
# Read data
dat = np.array(pd.read_csv(wk_dir + "Data/matern_02_1_1.csv", index_col=False, header = None))
original_dimension = (8000, 3, 100)
dat_full = dat.reshape(original_dimension)

plt.scatter(dat_full[:,0,0], dat_full[:,1,0], s = 20, c = dat_full[:,2,0])
# %%
alphas = [0, 0.025, 0.05, 0.1, 0.5, 1]# 0, 10, 100, 1000, 
iters = 50
MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
lr = 0.002 # default learning rate in keras adam
nnn = 5000 # Numbr of discrete grid of points to evaluate kde
lower = -800
upper = 800
KL_params = [nnn, lower, upper]
x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
theoretical_pdf = norm.pdf(x, 0, 202)
rho = 0.2
num_centers = [i**2 for i in range(10, 25, 4)] # 1104
# Number of layers and neurons
layers = 5
neurons = 50
eee = 1300
for i in range(iters):
    print(i)
    sub = dat_full[0:3000, :, i]
    X = sub[:, 0:2]
    Y = sub[:, 2]
    X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
    for idx, alpha in enumerate(alphas):
        model_1, density, W = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=eee, alpha = alpha,
                          device = device, n_centers=num_centers, 
                          theory = theoretical_pdf, rho = rho, kl_params = KL_params,
                          layers = layers, neurons=neurons)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE.iloc[i, idx] = model1_mse
toc = time.time()
MSE.to_csv(wk_dir + "HPC/" + "L5N50S3000_0211_bas1104.csv")
#d = np.exp(density)
#plt.plot(d)
#plt.title("Residual SPDE density for alpha=100000, iters = 3500, mse=0.22")

#www = W.cpu().detach().numpy()
#plt.hist(www)

# %%Process 2-6, large sample
# dat = np.array(pd.read_csv(wk_dir + "Data/process_2.csv", index_col=False, header = None))
# original_dimension = (2000, 3, 100)
# dat_full = dat.reshape(original_dimension)

# eee = 2500
# alphas = [0, 10, 100, 1000, 10000]
# iters = 40
# MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
# lr = 0.003 # default learning rate in keras adam
# nnn = 5000 # Numbr of discrete grid of points to evaluate kde
# lower = -800
# upper = 800
# x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
# theoretical_pdf = norm.pdf(x, 0, 202)
# for i in range(iters):
#     print(i)
#     sub = dat_full[0:1300, :, i]
#     X = sub[:, 0:2]
#     Y = sub[:, 2]
#     X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
#     for idx, alpha in enumerate(alphas):
#         #print(alpha)
#         model_1, density, W = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=eee, alpha = alpha,
#                           device = device, centers=fixed_centers, dims = out_dim, theory = theoretical_pdf)
#         X_test_tc = torch.tensor(X_test).float().to(device)
#         y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
#         model1_mse = np.mean((y_test - y0_model1)**2)
#         MSE.iloc[i, idx] = model1_mse
# MSE.to_csv(wk_dir + "Output_correct/Process2_1300sample.csv")


# # Process 3
# dat = np.array(pd.read_csv(wk_dir + "Data/process_3.csv", index_col=False, header = None))
# original_dimension = (2000, 3, 100)
# dat_full = dat.reshape(original_dimension)

# MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
# lr = 0.003 # default learning rate in keras adam
# nnn = 50000 # Numbr of discrete grid of points to evaluate kde
# lower = -800
# upper = 800
# x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
# theoretical_pdf = norm.pdf(x, 0, 202)
# for i in range(iters):
#     print(i)
#     sub = dat_full[0:1300, :, i]
#     X = sub[:, 0:2]
#     Y = sub[:, 2]
#     X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
#     for idx, alpha in enumerate(alphas):
#         #print(alpha)
#         model_1, density, W = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=eee, alpha = alpha,
#                           device = device, centers=fixed_centers, dims = out_dim, theory = theoretical_pdf)
#         X_test_tc = torch.tensor(X_test).float().to(device)
#         y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
#         model1_mse = np.mean((y_test - y0_model1)**2)
#         MSE.iloc[i, idx] = model1_mse
# MSE.to_csv(wk_dir + "Output_correct/Process3_1300sample.csv")

# # Process 4
# dat = np.array(pd.read_csv(wk_dir + "Data/process_4.csv", index_col=False, header = None))
# original_dimension = (2000, 3, 100)
# dat_full = dat.reshape(original_dimension)

# MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
# lr = 0.003 # default learning rate in keras adam
# nnn = 50000 # Numbr of discrete grid of points to evaluate kde
# lower = -800
# upper = 800
# x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
# theoretical_pdf = norm.pdf(x, 0, 202)
# for i in range(iters):
#     print(i)
#     sub = dat_full[0:1300, :, i]
#     X = sub[:, 0:2]
#     Y = sub[:, 2]
#     X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
#     for idx, alpha in enumerate(alphas):
#         #print(alpha)
#         model_1, density, W = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=eee, alpha = alpha,
#                           device = device, centers=fixed_centers, dims = out_dim, theory = theoretical_pdf)
#         X_test_tc = torch.tensor(X_test).float().to(device)
#         y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
#         model1_mse = np.mean((y_test - y0_model1)**2)
#         MSE.iloc[i, idx] = model1_mse
# MSE.to_csv(wk_dir + "Output_correct/Process4_1300sample.csv")

# # Process 6
# dat = np.array(pd.read_csv(wk_dir + "Data/process_6.csv", index_col=False, header = None))
# original_dimension = (2000, 3, 100)
# dat_full = dat.reshape(original_dimension)

# MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
# lr = 0.003 # default learning rate in keras adam
# nnn = 50000 # Numbr of discrete grid of points to evaluate kde
# lower = -800
# upper = 800
# x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
# theoretical_pdf = norm.pdf(x, 0, 202)
# for i in range(iters):
#     print(i)
#     sub = dat_full[0:1300, :, i]
#     X = sub[:, 0:2]
#     Y = sub[:, 2]
#     X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
#     for idx, alpha in enumerate(alphas):
#         #print(alpha)
#         model_1, density, W = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=eee, alpha = alpha,
#                           device = device, centers=fixed_centers, dims = out_dim, theory = theoretical_pdf)
#         X_test_tc = torch.tensor(X_test).float().to(device)
#         y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
#         model1_mse = np.mean((y_test - y0_model1)**2)
#         MSE.iloc[i, idx] = model1_mse
# MSE.to_csv(wk_dir + "Output_correct/Process6_1300sample.csv")


# # Process 2-6, small sample
# dat = np.array(pd.read_csv(wk_dir + "Data/process_2.csv", index_col=False, header = None))
# original_dimension = (2000, 3, 100)
# dat_full = dat.reshape(original_dimension)
# sample_size = 500
# eee = 2500
# alphas = [0, 10, 100, 1000, 10000]
# iters = 60
# MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
# lr = 0.003 # default learning rate in keras adam
# nnn = 50000 # Numbr of discrete grid of points to evaluate kde
# lower = -800
# upper = 800
# x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
# theoretical_pdf = norm.pdf(x, 0, 202)
# for i in range(iters):
#     print(i)
#     sub = dat_full[0:sample_size, :, i]
#     X = sub[:, 0:2]
#     Y = sub[:, 2]
#     X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
#     for idx, alpha in enumerate(alphas):
#         #print(alpha)
#         model_1, density, W = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=eee, alpha = alpha,
#                           device = device, centers=fixed_centers, dims = out_dim, theory = theoretical_pdf)
#         X_test_tc = torch.tensor(X_test).float().to(device)
#         y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
#         model1_mse = np.mean((y_test - y0_model1)**2)
#         MSE.iloc[i, idx] = model1_mse
# MSE.to_csv(wk_dir + "Output_correct/Process2_500sample.csv")


# # Process 3
# dat = np.array(pd.read_csv(wk_dir + "Data/process_3.csv", index_col=False, header = None))
# original_dimension = (2000, 3, 100)
# dat_full = dat.reshape(original_dimension)

# MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
# lr = 0.003 # default learning rate in keras adam
# nnn = 50000 # Numbr of discrete grid of points to evaluate kde
# lower = -800
# upper = 800
# x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
# theoretical_pdf = norm.pdf(x, 0, 202)
# for i in range(iters):
#     print(i)
#     sub = dat_full[0:sample_size, :, i]
#     X = sub[:, 0:2]
#     Y = sub[:, 2]
#     X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
#     for idx, alpha in enumerate(alphas):
#         #print(alpha)
#         model_1, density, W = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=eee, alpha = alpha,
#                           device = device, centers=fixed_centers, dims = out_dim, theory = theoretical_pdf)
#         X_test_tc = torch.tensor(X_test).float().to(device)
#         y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
#         model1_mse = np.mean((y_test - y0_model1)**2)
#         MSE.iloc[i, idx] = model1_mse
# MSE.to_csv(wk_dir + "Output_correct/Process3_500sample.csv")

# # Process 4
# dat = np.array(pd.read_csv(wk_dir + "Data/process_4.csv", index_col=False, header = None))
# original_dimension = (2000, 3, 100)
# dat_full = dat.reshape(original_dimension)

# MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
# lr = 0.003 # default learning rate in keras adam
# nnn = 50000 # Numbr of discrete grid of points to evaluate kde
# lower = -800
# upper = 800
# x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
# theoretical_pdf = norm.pdf(x, 0, 202)
# for i in range(iters):
#     print(i)
#     sub = dat_full[0:sample_size, :, i]
#     X = sub[:, 0:2]
#     Y = sub[:, 2]
#     X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
#     for idx, alpha in enumerate(alphas):
#         #print(alpha)
#         model_1, density, W = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=eee, alpha = alpha,
#                           device = device, centers=fixed_centers, dims = out_dim, theory = theoretical_pdf)
#         X_test_tc = torch.tensor(X_test).float().to(device)
#         y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
#         model1_mse = np.mean((y_test - y0_model1)**2)
#         MSE.iloc[i, idx] = model1_mse
# MSE.to_csv(wk_dir + "Output_correct/Process4_500sample.csv")

# # Process 6
# dat = np.array(pd.read_csv(wk_dir + "Data/process_6.csv", index_col=False, header = None))
# original_dimension = (2000, 3, 100)
# dat_full = dat.reshape(original_dimension)

# MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
# lr = 0.003 # default learning rate in keras adam
# nnn = 50000 # Numbr of discrete grid of points to evaluate kde
# lower = -800
# upper = 800
# x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
# theoretical_pdf = norm.pdf(x, 0, 202)
# for i in range(iters):
#     print(i)
#     sub = dat_full[0:sample_size, :, i]
#     X = sub[:, 0:2]
#     Y = sub[:, 2]
#     X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
#     for idx, alpha in enumerate(alphas):
#         #print(alpha)
#         model_1, density, W = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=eee, alpha = alpha,
#                           device = device, centers=fixed_centers, dims = out_dim, theory = theoretical_pdf)
#         X_test_tc = torch.tensor(X_test).float().to(device)
#         y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
#         model1_mse = np.mean((y_test - y0_model1)**2)
#         MSE.iloc[i, idx] = model1_mse
# MSE.to_csv(wk_dir + "Output_correct/Process6_500sample.csv")

# # %%Visualize train and tes
# plt.subplot(1,2,1)
# plt.scatter(X_test[:,0], X_test[:,1], s = 20, c = y_test)
# plt.title("Test true data")
# plt.subplot(1,2,2)
# plt.scatter(X_test[:,0], X_test[:,1], s = 20, c = y0_model1)
# plt.title("Test predicted data alpha=0.5, longer chain, more basis")
# #MSE.to_csv(wk_dir + "Output_New/Matern.csv")

# # %% Data with higher variance
# dat = np.array(pd.read_csv(wk_dir + "Data/matern_02_1_5.csv", index_col=False, header = None))
# size = 1300 * 3
# subdat = dat[0:size, :]
# original_dimension = (1300, 3, 100)
# dat_full = subdat.reshape(original_dimension)
# alphas = [3, 5, 7]#,0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 5, 10]
# iters = 30
# MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = alphas)
# lr = 0.0008 # default learning rate in keras adam
# for i in range(iters):
#     print(i)
#     sub = dat_full[:, :, i]
#     X = sub[:, 0:2]
#     Y = sub[:, 2]
#     X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
#     for idx, alpha in enumerate(alphas):
#         #print(alpha)
#         model_1 = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=1700, alpha = alpha,
#                           device = device, centers=fixed_centers, dims = out_dim)
#         X_test_tc = torch.tensor(X_test).float().to(device)
#         y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
#         model1_mse = np.mean((y_test - y0_model1)**2)
#         MSE.iloc[i, idx] = model1_mse
# MSE.to_csv(wk_dir + "Output_New/Matern_highV_sup.csv")
# # %% 


