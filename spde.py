
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
import copy
if torch.cuda.is_available():
    device = torch.device('cuda:0')
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

# Randomly split data and include validation set
# Train:Val:Test = 6:2:2
def random_split_val(X, Y):
    n = X.shape[0]
    rows = [i for i in range(0, n)]
    train_size = math.floor(n * 0.6)
    train_row = random.sample(range(n), train_size)
    train_row.sort()
    val_test_row = list(set(rows) - set(train_row))
    n_remain = len(val_test_row)
    val_size = math.floor(n_remain * 0.5)
    val_row = random.sample(val_test_row, val_size)
    val_row.sort()
    test_row = list(set(val_test_row) - set(val_row))
    X_train = X[train_row, :]
    y_train = Y[train_row]
    X_val = X[val_row, :]
    y_val = Y[val_row]
    X_test = X[test_row, :]
    y_test = Y[test_row]
    return X_train, X_val, X_test, y_train, y_val, y_test


# Out of support splitting
def out_support_split(X, Y):
    x_row = np.logical_and(X[:, 0] < 0.9, X[:, 0] > 0.1)
    y_row = np.logical_and(X[:, 1] < 0.9, X[:, 1] > 0.1)
    train_row = np.logical_and(x_row, y_row)
    test_row = np.invert(train_row)
    X_train = X[train_row, :]
    X_test  = X[test_row, :]
    y_train = Y[train_row]
    y_test = Y[test_row]
    return X_train, X_test, y_train, y_test


# Function to update the loss plot
def live_plot(loss, kl, total_loss, val_loss):
    clear_output(wait=True)
    plt.subplot(1,3,1)
    plt.plot(loss, label = "Training loss")
    plt.plot(val_loss, label = "Validation loss")
    plt.title('MSE Loss')
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


class RBF(nn.Module):
    def __init__(self, n_centers):
        super(RBF, self).__init__()
        self.out_features = np.sum(n_centers)
        self.num_basis = n_centers
        self.knots_1d = [torch.linspace(0,1,int(np.sqrt(i))) for i in n_centers]
        center = torch.zeros((self.out_features, 2))
        sum = 0
        for i in range(len(n_centers)):
            amount = n_centers[i]
            knots_s1, knots_s2 = torch.meshgrid(self.knots_1d[i],self.knots_1d[i])
            knots = torch.column_stack((knots_s1.flatten(),knots_s2.flatten()))
            center[sum:sum+amount, :] = knots
            sum += amount
        self.register_buffer("centers", center)

    def forward(self, x):
        # The dimension of output should be x.size(0) * self.out_features
        # Input: X is the coordinate of size N * 2
        self.centers.to(device)
        size = (x.size(0), self.out_features, x.size(1))
        x = x.unsqueeze(1).expand(size)
        centers = self.centers.unsqueeze(0).expand(size)
        distances = torch.norm(x - centers, dim=2)
        '''
        # Divide distance by theta
        res = len(self.num_basis)
        K = 0
        for i in range(res):
            theta = 1/np.sqrt(self.num_basis[i])*2.5
            base = self.num_basis[i]
            distances[:, K:K+base] = distances[:, K:K+base] / theta
            K += base
        '''
        # Applying the specified formula
        rbf_output = (1 - distances)**6 * (35 * distances**2 + 18 * distances + 3) / 3
        return rbf_output



'''
        out = torch.zeros((x.shape[0], self.out_features)).to(device)
        N = x.shape[0] # Input sample size
        K = 0 # Basis size
        for i in range(len(self.num_basis)):
            theta = 1/np.sqrt(self.num_basis[i])*2.5 # This is the average distance * 2.5
            for j in range(self.num_basis[i]):
                d = torch.norm(x-self.centers[i, :], dim = 1)/theta
                d = d.reshape(-1)
                for k in range(len(d)):
                    out[k,j + K] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
            K += self.num_basis[i]
        return out
'''
class RBFNetwork(nn.Module):
    def __init__(self, n_centers, n_layers=5, n_neurons=100):
        super(RBFNetwork, self).__init__()
        self.rbf_layer = RBF(n_centers=n_centers)  # Assuming RBF is defined elsewhere
        out_dim = np.sum(n_centers)  # Total output dimension from RBF layer

        # First fully connected layer after RBF output
        self.fc_hidden_layer = nn.Linear(out_dim, n_neurons)
        
        # Dynamically create the hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(n_neurons, n_neurons) for _ in range(n_layers)])
        
        # Output layer
        self.output_layer = nn.Linear(n_neurons, 1)

    def forward(self, x):
        # Forward pass through RBF layer
        x = self.rbf_layer(x)
        
        # Activation function after first fully connected layer
        x = F.relu(self.fc_hidden_layer(x))
        
        # Forward pass through all hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        # Output layer
        x = self.output_layer(x)
        return x

class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()

        # Extract layers from the original model
        self.rbf_layer = original_model.rbf_layer
        self.fc_hidden_layer = original_model.fc_hidden_layer
        self.hidden_layers = original_model.hidden_layers
        
        # Freeze parameters in extracted layers
        for param in self.rbf_layer.parameters():
            param.requires_grad = False
        for param in self.fc_hidden_layer.parameters():
            param.requires_grad = False
            for layer in self.hidden_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

    
        # Define 5 new fully connected layers
        self.fc1 = nn.Linear(100, 100)  # Adjust output sizes as needed
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 1)  

    def forward(self, x):
        x = self.rbf_layer(x)
        x = self.fc_hidden_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)

        x = nn.functional.relu(self.fc1(x))  
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = self.fc5(x)  # No activation on final layer for regression
        return x


def gauss_kde(data, lower, upper, n, bw = None):
    """
        Gaussian kernel density estimation
        data: input data
        lower, upper: lower and upper bound of the input data
        bw: bandwidth in estimating the density
    """
    x = torch.ravel(data)
    grid = torch.linspace(lower, upper, n, device=x.device)
    # default bw: 1.06 * std * n^(-1/5)
    std = torch.std(x)
    if bw is None:
        bw = len(x)**(-1 / 5) * std * 1.06
    norm_factor = (2 * np.pi)**0.5 * len(x) * bw
    out = torch.sum(
        torch.exp(
            -0.5 * torch.square(
                (x[:, None] - grid[None, :]) / bw
            )
        ),
        axis=0
    ) / norm_factor
    return out


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                                   Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



# RBF loss function
def RBF_loss_func(X, y_true, model, optimizer, alpha, device, theory_pdf, rho, kl):
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
    nnn = kl[0] # Numbr of discrete grid of points to evaluate kde
    lower = kl[1]
    upper = kl[2]
    epsilon = 1e-8  # A small value to ensure numerical stability
    W = ((8.0**0.5) / rho)**2 * y_pred - out
    W_density = gauss_kde(W, lower, upper, nnn)
    W_density = torch.maximum(W_density, torch.tensor(epsilon))
    # Plot density
    W_density = torch.log(W_density)
    #print(W_density)
    #x = np.linspace(lower, upper, nnn) # Define the range over which to evaluate the KDE and theoretical PDF
    theoretical_pdf = theory_pdf
    
    theoretical_pdf = np.maximum(theoretical_pdf, epsilon) # Take the exponential of target
    theoretical_tc = torch.tensor(theoretical_pdf).to(device)
    # Input should be in log space
    kl_loss = nn.KLDivLoss(reduction="batchmean") # 1) By default, log_target = false 2) sum of pointwise loss
    PINN = kl_loss(W_density, theoretical_tc) # input, log_target

    alpha = torch.tensor(alpha).to(device)
    loss = mse_loss + alpha * PINN
    kl_divergence = PINN.cpu().detach().numpy()
    loss.backward()
    return mse_loss, kl_divergence, loss, W_density, W



# RBF Training
def RBF_train(X_train, X_val, y_train, y_val, lr, epochs, alpha, device, n_centers, theory, rho, kl_params, layers, neurons, raw_model = None):
    y_train = y_train.reshape(-1, 1)
    X_val_tc = torch.tensor(X_val).float().to(device)
    y_val_tc = torch.tensor(y_val).float().to(device)
    # Initialize model, loss, and optimizer
    y_train_tc = torch.from_numpy(y_train).float().to(device)
    X_train_tc = torch.tensor(X_train, requires_grad=True).float().to(device)
    if raw_model == None:
        model = RBFNetwork(n_centers=n_centers, n_layers = layers, n_neurons = neurons).to(device)
    else:
        model = ModifiedModel(raw_model)
    # optimizer = torch.optim.SGD(model.parameters(), lr=2e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #early_stopping = EarlyStopping(patience=10, verbose=False, path='model_checkpoint.pt')
    model.train()
    loss_values = []
    val_loss = []
    kl_values = []
    total_values = []
    for i in range(epochs):
        mse_loss, kl, total_loss, density, W = RBF_loss_func(X_train_tc, y_train_tc, model, optimizer, alpha, device, theory, rho, kl_params)
        optimizer.step()
        # Early stopping

        with torch.no_grad():
            output = model(X_val_tc)

        # Plotting
        loss_values.append(mse_loss.cpu().detach().numpy())
        total_values.append(total_loss.cpu().detach().numpy())
        kl_values.append(kl)
        y_val_hat = output.cpu().detach().numpy().reshape(-1)
        val_mse = np.mean((y_val - y_val_hat)**2)
        val_loss.append(val_mse)
        #live_plot(loss_values, kl_values, total_values, val_loss)
    density = density.cpu().detach().numpy()
    #model.load_state_dict(torch.load('model_checkpoint.pt'))
    return model, density, W



# Kriging prediction
# Return: predicted value
# Assumes that X_train and X_test only has coordinates
def Kriging(X_train, X_test, y_train, spatial_corr):
    N = X_train.shape[0] + X_test.shape[0]
    n2 = X_test.shape[0] # Test size
    s_train = X_train
    s_test = X_test
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



#%% RBF with random centers
class RBF_random(nn.Module):
    def __init__(self, in_features, out_features):
        super(RBF_random, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weights = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.centers, -1, 1)
        nn.init.uniform_(self.weights, -1, 1)

    def forward(self, x):
        size = (x.size(0), self.out_features, self.in_features)
        x = x.unsqueeze(1).expand(size)
        centers = self.centers.unsqueeze(0).expand(size)
        distances = torch.norm(x - centers, dim=2)
        rbf_output = (1 - distances)**6 * (35 * distances**2 + 18 * distances + 3) / 3
        weighted_output = rbf_output * self.weights
        return weighted_output
    
class RBFNetworkRandom(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(RBFNetworkRandom, self).__init__()
        self.rbf_layer = RBF_random(in_features = in_dims, out_features=out_dims)
        self.fc_hidden_layer = nn.Linear(out_dims, 100)  # 100 is an example size for the hidden FC layer
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

# RBF Training
def RBFRandmTrain(X_train, y_train, lr, epochs, alpha, device, in_dims, out_dims):
    # Train-test split
    y_train = y_train.reshape(-1, 1)
    # Initialize model, loss, and optimizer
    y_train_tc = torch.from_numpy(y_train).float().to(device)
    X_train_tc = torch.tensor(X_train, requires_grad=True).float().to(device)
    model = RBFNetworkRandom(in_dims = in_dims, out_dims=out_dims).to(device)
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

