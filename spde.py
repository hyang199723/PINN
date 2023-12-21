
#%% Utility function to generate data, etc
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
from IPython.display import clear_output
import scipy
from scipy.stats import gaussian_kde, norm
from scipy.special import gamma, kv
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
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
def gen_stat(N, rho, spatial_var, noise_var):
    n = N
    random.seed(123)
    length = 1
    coords1 = np.random.uniform(0, length, n)
    coords2 = np.random.uniform(0, length, n)
    coords = np.vstack((coords1, coords2)).T
    X = np.zeros((n, 3))
    X[:, 0] = 1
    X[:, 1] = coords1
    X[:, 2] = coords2

    # Exponential Correlation
    distance = distance_matrix(coords.reshape(-1, 2), coords.reshape(-1, 2))
    corr = np.exp(-distance / rho)
    # Cholesky decomposition and generate correlated data
    L = np.linalg.cholesky(spatial_var*corr)
    z = np.random.normal(0, 1, n)
    Y = np.dot(L, z) + np.random.normal(0, noise_var, n)
    return X, Y





# Feedford network without penalty
class FeedForwardNN(nn.Module):
    def __init__(self, layers):
        super(FeedForwardNN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = torch.nn.ReLU
        
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
def model_train_dnn(X_train, y_train, layers, lr, epochs, device):
    # Train-test split
    y_train = y_train.reshape(-1, 1)
    # Initialize model, loss, and optimizer
    X_train_tc = torch.from_numpy(X_train).float().to(device)
    y_train_tc = torch.from_numpy(y_train).float().to(device)
    
    layers = layers
    model = FeedForwardNN(layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3 / 200)
    model.train()
    for epoch in range(epochs):
        y_pred = model(X_train_tc)
        
        loss = criterion(y_pred, y_train_tc)
        
        #print(loss)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
    return model

# Function to update the loss plot
def live_plot(loss, kl, total_loss):
    clear_output(wait=True)
    plt.subplot(1,3,1)
    plt.plot(loss)
    plt.title('Training Loss')
    plt.xlabel('Batch number')
    plt.ylabel('Loss')
    plt.subplot(1,3,2)
    plt.plot(kl)
    plt.title('KL divergence')
    plt.xlabel('Batch number')
    plt.ylabel('KL')
    plt.subplot(1,3,3)
    plt.plot(total_loss)
    plt.title('Total loss')
    plt.xlabel('Batch number')
    plt.ylabel('Total')
    plt.show()


# Second order loss function SPDE
def loss_func(X, y_true, model, optimizer, alpha, device):
    optimizer.zero_grad()
    dims = X.shape[1]
    x0 = X[:, 0].reshape(-1 ,1)
    x1 = X[:, 1].reshape(-1 ,1)
    x2 = X[:, 2].reshape(-1 ,1)
    XX = torch.cat((x0, x1, x2, X[:, 3:dims]), dim = 1).to(device)
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
    # kappa should be half of the range
    W_init = 0.1/2 * y_pred - out
    PINN = torch.mean(W_init)
    W = W_init.cpu().detach().numpy()
    W = W.reshape(-1)
    # KDE 
    kde = gaussian_kde(W)
    x = np.linspace(min(W), max(W), 1000) # Define the range over which to evaluate the KDE and theoretical PDF
    empirical_pdf = kde(x) # Evaluate the estimated empirical PDF
    theoretical_pdf = norm.pdf(x, 0, 1 + 0.1) 
    epsilon = 1e-10  # A small value to ensure numerical stability
    empirical_pdf = np.maximum(empirical_pdf, epsilon)
    empirical_tc = torch.tensor(empirical_pdf, requires_grad=True) # Convert to torch object
    theoretical_pdf = np.maximum(theoretical_pdf, epsilon)
    theoretical_tc = torch.tensor(theoretical_pdf, requires_grad=True)
    y = empirical_tc * torch.log(empirical_tc / theoretical_tc)
    x = torch.tensor(x)
    kl = torch.trapezoid(y, x)
    kl = torch.tensor(kl, requires_grad=True).float().to(device)
    kl_divergence = np.trapz(empirical_pdf * np.log(empirical_pdf / theoretical_pdf), x)

    SINN = alpha * kl
    SINN = torch.tensor(SINN, requires_grad=True).float().to(device)
    loss = mse_loss + PINN
    loss.backward()
    return mse_loss, kl_divergence, loss

# PINN
# Return the predicted values
def model_train_pinn(X_train, y_train, layers, lr, epochs, alpha, momentum, device):
    # Train-test split
    y_train = y_train.reshape(-1, 1)
    # Initialize model, loss, and optimizer
    y_train_tc = torch.from_numpy(y_train).float().to(device)
    X_train_tc = torch.tensor(X_train, requires_grad=True).float().to(device)
    layers = layers
    model = FeedForwardNN(layers).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=2e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3 / 200)
    model.train()
    loss_values = []
    kl_values = []
    total_values = []
    for epoch in range(epochs):
        mse_loss, kl, total_loss = loss_func(X_train_tc, y_train_tc, model, optimizer, alpha, device)
        loss_values.append(mse_loss.cpu().detach().numpy())
        total_values.append(total_loss.cpu().detach().numpy())
        kl_values.append(kl)
        live_plot(loss_values, kl_values, total_values)
        optimizer.step()
    return model

class RBF(nn.Module):
    def __init__(self, out_features, fixed_centers):
        super(RBF, self).__init__()
        self.out_features = out_features
        self.register_buffer('centers', fixed_centers)
        self.weights = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weights, -1, 1)

    def forward(self, x):
        size = (x.size(0), self.out_features, x.size(1))
        x = x.unsqueeze(1).expand(size)
        centers = self.centers.unsqueeze(0).expand(size)
        distances = torch.norm(x - centers, dim=2)
        # Applying the specified formula
        rbf_output = (1 - distances)**6 * (35 * distances**2 + 18 * distances + 3) / 3
        # Multiply by weights
        weighted_output = rbf_output * self.weights
        return weighted_output


class RBFNetwork(nn.Module):
    def __init__(self, fixed_centers, out_dim):
        super(RBFNetwork, self).__init__()
        self.rbf_layer = RBF(out_features=out_dim, fixed_centers=fixed_centers)
        self.fc_hidden_layer = nn.Linear(out_dim, 100)  # 100 is an example size for the hidden FC layer
        self.hidden_layer_1 = nn.Linear(100, 100)
        self.hidden_layer_2 = nn.Linear(100, 100)
        self.hidden_layer_3 = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, 1)  # Assuming a single output neuron for simplicity

    def forward(self, x):
        x = self.rbf_layer(x)
        x = F.relu(self.fc_hidden_layer(x))
        x = F.relu(self.hidden_layer_1(x))
        x = F.relu(self.hidden_layer_2(x))
        x = F.relu(self.hidden_layer_3(x))
        x = self.output_layer(x)
        return x
    

# RBF loss function
def RBF_loss_func(X, y_true, model, optimizer, alpha, device):
    optimizer.zero_grad()
    x0 = X[:, 0].reshape(-1 ,1)
    x1 = X[:, 1].reshape(-1 ,1)
    XX = torch.cat((x0, x1), dim = 1).to(device)
    y_pred = model(XX)
    mse_loss = torch.mean((y_true - y_pred) ** 2) # MSE loss
    # Calculate gradient
    y_x = torch.autograd.grad(
            y_pred,x0, 
            grad_outputs=torch.ones_like(y_pred),
            retain_graph=True,
            create_graph=True
        )[0]
    y_xx = torch.autograd.grad(
            y_x, x0, 
            grad_outputs=torch.ones_like(y_x),
            retain_graph=True,
            create_graph=True
        )[0]
    y_z = torch.autograd.grad(
            y_pred, x1, 
            grad_outputs=torch.ones_like(y_pred),
            retain_graph=True,
            create_graph=True
        )[0]
    y_zz = torch.autograd.grad(
            y_z, x1, 
            grad_outputs=torch.ones_like(y_z),
            retain_graph=True,
            create_graph=True
        )[0]
    out = y_xx + y_zz
    # Second order deravative should follow normal distribution
    # kappa should be half of the range
    W = ((8.0**0.5) / 0.3)**2 * y_pred - out

    W = W.cpu().detach().numpy()
    W = W.reshape(-1)
    # KDE 
    kde = gaussian_kde(W)
    x = np.linspace(min(W), max(W), 1000) # Define the range over which to evaluate the KDE and theoretical PDF
    empirical_pdf = kde(x) # Evaluate the estimated empirical PDF
    theoretical_pdf = norm.pdf(x, 0, 1) 
    epsilon = 1e-10  # A small value to ensure numerical stability
    empirical_pdf = np.maximum(empirical_pdf, epsilon)
    empirical_tc = torch.tensor(empirical_pdf, requires_grad=True) # Convert to torch object
    theoretical_pdf = np.maximum(theoretical_pdf, epsilon)
    theoretical_tc = torch.tensor(theoretical_pdf, requires_grad=True)
    kl_loss = nn.KLDivLoss(reduction="mean")
    PINN = kl_loss(empirical_tc, theoretical_tc)

    alpha = torch.tensor(alpha)
    #PINN = torch.mean(W**2)
    kl_divergence = PINN.cpu().detach().numpy()
    loss = mse_loss + alpha * PINN
    loss.backward()
    return mse_loss, kl_divergence, loss


# RBF Training
def RBF_train(X_train, y_train, lr, epochs, alpha, device, centers, dims):
    # Train-test split
    y_train = y_train.reshape(-1, 1)
    # Initialize model, loss, and optimizer
    y_train_tc = torch.from_numpy(y_train).float().to(device)
    X_train_tc = torch.tensor(X_train, requires_grad=True).float().to(device)
    model = RBFNetwork(fixed_centers=centers, out_dim=dims).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=2e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    loss_values = []
    kl_values = []
    total_values = []
    for epoch in range(epochs):
        mse_loss, kl, total_loss = RBF_loss_func(X_train_tc, y_train_tc, model, optimizer, alpha, device)
        loss_values.append(mse_loss.cpu().detach().numpy())
        total_values.append(total_loss.cpu().detach().numpy())
        kl_values.append(kl)
        #live_plot(loss_values, kl_values, total_values)
        optimizer.step()
    return model

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

#%% Modified Bessel function of second kind and Matern correlation
# kappa = \sqrt(8v) / rho
# Distance: distances between observations
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


# Generate from a matern correlation process
# N: number of data points
# rho: spatial correlation
# vvv: variance
def gen_matern(N, rho, spatial_var, noise_var, nu):
    n = N
    random.seed(123)
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


# Generate the log-normal process
def gen_lognormal(N, rho, spatial_var, noise_var, nu):
    n = N
    random.seed(123)
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
    Y = np.exp(np.dot(L, z)) + np.random.normal(0, noise_var, n)
    return X, Y

# Generate non-stationary process from mixture
def gen_mixture(N, spatial_var, noise_var, nu1, nu2, rho1, rho2):
    length = 1
    coords = np.random.uniform(0, length, (N, 2))
    X = np.zeros((N, 3))
    X[:, 0] = 1
    X[:, 1:3] = coords
    # Compute distance matrix
    distance = distance_matrix(coords.reshape(-1, 2), coords.reshape(-1, 2))

    # Compute the Matern correlation matrices
    cor1 = Matern_Cor(nu1, rho1, distance)
    cor2 = Matern_Cor(nu2, rho2, distance)

    # Simulate the spatial process
    L1 = np.linalg.cholesky(spatial_var*cor1)
    L2 = np.linalg.cholesky(spatial_var*cor2)
    z1 = z2 = np.random.normal(0, 1, N)
    y1 = np.dot(L1, z1) + np.random.normal(0, noise_var, N)
    y2 = np.dot(L2, z2) + np.random.normal(0, noise_var, N)

    # Combine the two fields
    w = coords[:, 0] * 0.5
    y = y1 * w + y2 * (1 - w)
    return X, y