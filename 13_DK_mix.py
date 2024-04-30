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
from spde import *
import GPy

n = 50
N = 50 * 50
np.random.seed(123)
s_1 = np.linspace(0,1,50).reshape(-1,1)
s_2 = np.linspace(0,1,50).reshape(-1,1)
s_x,s_y=np.meshgrid(s_1,s_2)
s = np.vstack((s_x.flatten(),s_y.flatten())).T
pi = np.random.binomial(1,0.5,size=N)
y = 2 * np.sin( (s[:,0]+0.5)*(s[:,1] + 0.5)) * pi + (np.sin(s[:,0]*s[:,1]))*(1 - pi)
y = y + np.random.normal(0,0.2, size=N) * pi + (np.random.normal(0,0.2, size=N))*(1 - pi)
y = np.exp(y)
y = y.reshape(-1,1)

n = 50
N = 50 * 50
s_1 = np.linspace(0,1,50).reshape(-1,1)
s_2 = np.linspace(0,1,50).reshape(-1,1)
s_x,s_y=np.meshgrid(s_1,s_2)
s = np.vstack((s_x.flatten(),s_y.flatten())).T
kernel1 = GPy.kern.Exponential(2,3,0.2) ##Covariance Function
kernel2 = GPy.kern.Matern52(2,1,0.5) ##Covariance Function
cov_mat1 = kernel1.K(s)
cov_mat2 = kernel2.K(s)
# Generate M sample path with mean mu and covariance C
np.random.seed(123)
y1 = np.random.multivariate_normal(np.zeros(N),cov_mat1).T
y2 = np.random.multivariate_normal(np.ones(N),cov_mat2).T
pi = np.random.binomial(1,0.5,size=N)
y = y1 + y2*0.5
y = np.exp(y/3+1)
y = np.random.poisson(y)
y = y.reshape(-1,1)

##Visualization of one replicate
y_mat = y.reshape(n,n)
fig, ax = plt.subplots()
im = ax.imshow(y_mat, extent=[0, 1, 0, 1], origin="lower",
               vmax=y_mat.max(), vmin=y_mat.min())
plt.colorbar(im)
plt.show()
#%% 
#alphas = [0]# 0, 10, 100, 1000, 
# Number of layers and neurons
#layers = 3
neurons = 100
layers = [6]
iters = 2 # 10-fold CV
MSE = pd.DataFrame(data = 0.0, index = range(iters), columns = layers)
lr = 0.001 # default learning rate in keras adam
nnn = 5000 # Numbr of discrete grid of points to evaluate kde
lower = -800
upper = 800
KL_params = [nnn, lower, upper]
x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
theoretical_pdf = norm.pdf(x, 0, 202)
rho = 0.2
#num_centers = [10**2,19**2,37**2]
num_centers = [20**2,38**2,70**2]

eee = 1500 # NN iterations
X = s
Y = y
for i in range(iters):
    print(i)
    X_train, X_val, X_test, y_train, y_val, y_test = random_split_val(X, Y)
    for idx, layer in enumerate(layers):
        model_1, density, W = RBF_train(X_train, X_val, y_train, y_val, lr=lr, epochs=eee, alpha = 0,
                          device = device, n_centers=num_centers, 
                          theory = theoretical_pdf, rho = rho, kl_params = KL_params,
                          layers = layer, neurons=neurons)
        X_test_tc = torch.tensor(X_test).float().to(device)
        y0_model1 = model_1(X_test_tc).cpu().detach().numpy().reshape(-1)
        model1_mse = np.mean((y_test - y0_model1)**2)
        MSE.iloc[i, idx] = model1_mse
#MSE.to_csv(wk_dir + "Output_correct/DK_Compare_centers.csv")
# %%
