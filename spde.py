# Utility function to generate data, etc
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math
from scipy.spatial import distance_matrix
import torch.nn as nn
import torch.optim as optim
import scipy
from collections import OrderedDict
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

#%% Functions
# Simulate a random 2-D non-stationary data, same covariance across map
# N: number of data points
# rho: spatial correlation
# vvv: variance
def gen_non_same(N, rho, vvv, nugget):
    n = N
    random.seed(123)
    length = 20
    coords_x = np.random.uniform(0, length, n)
    coords_y = np.random.uniform(0, length, n)
    coords = np.zeros((n, 2))
    coords[:, 0] = coords_x
    coords[:, 1] = coords_y**2
    X = np.zeros((n, 3))
    X[:, 0] = 1
    X[:, 1] = coords_x
    X[:, 2] = coords_y

    # Exponential Correlation
    distance = distance_matrix(coords.reshape(-1, 2), coords.reshape(-1, 2))
    corr = np.exp(-distance / rho)
    # Cholesky decomposition and generate correlated data
    L = np.linalg.cholesky(vvv*corr)
    z = np.random.normal(0, nugget, n)
    Y = np.dot(L, z)
    return X, Y

# Simulate multi time steps non-stationary data
def gen_multiTS(N, rho, vvv, ts):
    Y_full = np.zeros((N, ts))
    length = 20
    coords_x = np.random.uniform(0, length, N)
    coords_y = np.random.uniform(0, length, N)
    coords = np.zeros((N, 2))
    coords[:, 0] = coords_x
    coords[:, 1] = coords_y**2
    X = np.zeros((N, 3))
    X[:, 0] = 1
    X[:, 1] = coords_x
    X[:, 2] = coords_y
    
    # Exponential Correlation
    distance = distance_matrix(coords.reshape(-1, 2), coords.reshape(-1, 2))
    corr = np.exp(-distance / rho)
    L = np.linalg.cholesky(vvv*corr)
    for i in range(0, ts):
        foo = np.random.normal(0, 1, N)
        temp = np.dot(L, foo)
        Y_full[:, i] = temp
    return X, Y_full

# Simulate stationary 2-D data
# N: number of data points
# rho: spatial correlation
# vvv: variance
def gen_stat(N, rho, vvv):
    n = N
    random.seed(123)
    length = 20
    coords1 = np.random.uniform(0, length, n)
    coords2 = np.random.uniform(0, length, n)
    coords = np.concatenate((coords1, coords2), 1)
    X = np.zeros((n, 3))
    X[:, 0] = 1
    X[:, 1] = coords
    X[:, 2] = coords2

    # Exponential Correlation
    distance = distance_matrix(coords.reshape(-1, 2), coords.reshape(-1, 2))
    corr = np.exp(-distance / rho)
    # Cholesky decomposition and generate correlated data
    L = np.linalg.cholesky(vvv*corr)
    z = np.random.normal(0, 1, n)
    Y = np.dot(L, z)
    return X, Y

# Feedford network without penalty
class FeedForwardNN(nn.Module):
    def __init__(self, layers):
        super(FeedForwardNN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = torch.nn.Tanh
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, x):
        out = self.layers(x)
        return out


# Randomly split data
def random_split(X, Y):
    n = X.shape[0]
    rows = [i for i in range(0, n)]
    train_size = math.floor(n * 0.8)
    random.seed(123)
    train_row = random.sample(range(n), train_size)
    train_row.sort()
    test_row = list(set(rows) - set(train_row))
    X_train = X[train_row, :]
    y_train = Y[train_row]
    X_test = X[test_row, :]
    y_test = Y[test_row]
    return X_train, X_test, y_train, y_test

# Out of support splitting
def out_support_split(X, Y):
    train_row = X[:, 1] > 17.5
    test_row = np.invert(train_row)
    X_train = X[train_row, :]
    X_test  = X[test_row, :]
    y_train = Y[train_row]
    y_test = Y[test_row]
    return X_train, X_test, y_train, y_test


# Deep neural network without PINN
# Layers: A layer specifying the NN dimensions
def model_train_dnn(X, Y, X_test, y_test, layers, lr, epochs, split = "random"):
    # Train-test split
    Y = Y.reshape(-1, 1)
    if split == "random":
        X_train, X_test, y_train, y_test = random_split(X, Y)
    else:
        X_train, X_test, y_train, y_test = out_support_split(X, Y)
    
    # Initialize model, loss, and optimizer
    X_train_tc = torch.from_numpy(X_train).to(torch.float32)
    y_train_tc = torch.from_numpy(y_train).to(torch.float32)
    
    layers = layers
    model = FeedForwardNN(layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        y_pred = model(X_train_tc)
        
        loss = criterion(y_pred, y_train_tc)
        
        #print(loss)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
    return model

# Second order loss function SPDE
def loss_func(X, y_true, model, optimizer, alpha, device):
    x0 = X[:, 0].reshape(-1 ,1)
    x1 = X[:, 1].reshape(-1 ,1)
    x2 = X[:, 2].reshape(-1 ,1)
    optimizer.zero_grad()
    XX = torch.cat((x0, x1, x2), dim = 1).to(device)
    y_pred = model(XX)
    mse_loss = torch.mean((y_true - y_pred) ** 2) # MSE loss
    # Calculate gradient
    y_x = torch.autograd.grad(
            y_pred,x1, 
            grad_outputs=torch.ones_like(y_pred),
            retain_graph=True,
            create_graph=True
        )[0]
    y_xx = torch.autograd.grad(
            y_x, x1, 
            grad_outputs=torch.ones_like(y_x),
            retain_graph=True,
            create_graph=True
        )[0]
    y_z = torch.autograd.grad(
            y_pred, x2, 
            grad_outputs=torch.ones_like(y_pred),
            retain_graph=True,
            create_graph=True
        )[0]
    y_zz = torch.autograd.grad(
            y_z, x2, 
            grad_outputs=torch.ones_like(y_z),
            retain_graph=True,
            create_graph=True
        )[0]
    out = y_xx + y_zz
    # Second order deravative should follow normal distribution
    # kappa?
    # Convert range to kappa
    # kappa should be half of the range
    W = 3/2 * y_pred - out
    # KDE 
    kde = gauss_kde(W, -10, 10, 10000)
    # KL_divergence
    kde = kde.cpu().detach().numpy()
    kurtosis = scipy.stats.kurtosis(kde)
    skew = scipy.stats.skew(kde)
    mean = np.mean(kde)
   # W = out
    #target = np.random.normal(size = len(out))
    #gauss_kde(t, -5,5,100)
    #f = torch.mean(W ** 2)
    loss = mse_loss + alpha * (skew + mean + kurtosis)# Weight 
    # print(loss) 
    loss.backward()
    return loss

# PINN
# Return the predicted values
def model_train_pinn(X_train, y_train, layers, lr, epochs, alpha, device):
    # Train-test split
    y_train = y_train.reshape(-1, 1)
    # Initialize model, loss, and optimizer
    y_train_tc = torch.from_numpy(y_train).float().to(device)
    X_train_tc = torch.tensor(X_train, requires_grad=True).float().to(device)
    
    
    layers = layers
    model = FeedForwardNN(layers).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=2e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        loss_func(X_train_tc, y_train_tc, model, optimizer, alpha, device)
        optimizer.step()
    return model

# kde
def gauss_kde(x, lower, upper, n, bw=None):
    x = torch.ravel(x)
    grid = torch.linspace(lower, upper, n, device=x.device)
    if bw is None:
        bw = len(x)**(-1 / 5)
    norm_factor = (2 * np.pi)**0.5 * len(x) * bw
    return torch.sum(
        torch.exp(
            -0.5 * torch.square(
                (x[:, None] - grid[None, :]) / bw
            )
        ),
        axis=0
    ) / norm_factor

# Kriging prediction
# Return: predicted value
def Kriging(X_train, X_test, y_train, N, v, spatial_corr):
    n2 = X_test.shape[0] # Test size
    s_train = X_train[:, 1:3]
    s_test = X_test[:, 1:3]
    coords = np.concatenate((s_test, s_train))
    distance = distance_matrix(coords.reshape(-1, 2), coords.reshape(-1, 2))
    v_bar = np.var(y_train)
    cov_bar = v_bar * np.exp(-distance / spatial_corr)

    sigma11 = cov_bar[0:n2, 0:n2]
    sigma12 = cov_bar[0:n2, n2:N]
    sigma21 = np.transpose(sigma12)
    sigma22 = cov_bar[n2:N, n2:N]

    sigma22_inv = np.linalg.inv(sigma22)
    sigma_bar = sigma11 - np.dot(np.dot(sigma12, sigma22_inv), sigma21)

    mu_bar = np.dot(np.dot(sigma12, sigma22_inv), y_train)

    zzz = np.random.normal(0, 1, n2)
    L = np.linalg.cholesky(sigma_bar)
    y_hat = mu_bar + np.dot(L, zzz)
    return y_hat