# Verify the Equation (2) in SPDE paper
# https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/j.1467-9868.2011.00777.x
# (k^2 - delta)^{alpha / 2} * x(u) = white noise process

#%% Packages
import scipy
from scipy.stats import gaussian_kde, norm
from scipy.special import gamma, kn, kv
import numpy as np
import random
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import skgstat as skg

#%% Generate Matern process
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

def gen_expo(N, rho, spatial_var, noise_var):
    n = N
    random.seed(123)
    length = 1
    coords = np.random.uniform(0, length, (N, 2))
    X = np.zeros((n, 3))
    X[:, 0] = 1
    X[:, 1:3] = coords

    # Exponential Correlation
    distance = distance_matrix(coords.reshape(-1, 2), coords.reshape(-1, 2))
    corr = np.exp(-distance / rho)
    # Cholesky decomposition and generate correlated data
    L = np.linalg.cholesky(spatial_var*corr)
    z = np.random.normal(0, 1, n)
    Y = np.dot(L, z) + np.random.normal(0, noise_var, n)
    return X, Y

# Single value matern correlation
def matern_correlation(nu, rho, distance):
    # Scaling factor for the distance
    scaled_distance = np.sqrt(8 * nu) * distance / rho
    
    # Calculate the correlation using the Matérn function
    correlation = (2**(1-nu) / gamma(nu)) * (scaled_distance)**nu * kv(nu, scaled_distance)
    
    return correlation

#%% Verify paper claim
nus = np.linspace(0.5, 10, 100)
cors = np.zeros(100)
for i, nu in enumerate(nus):
    cors[i] = matern_correlation(nu, 0.1, 0.1)

plt.plot(nus, cors)

#%%
N = 1000
P = 2
noise_var = 0.1
rho = 0.2
nu = 1
kappa = (8 * nu)**(0.5) / rho
spatial_var = 1
X, Y = gen_matern(N, rho, spatial_var, noise_var, nu)
X = X[:, 1:3] # Only need coors

V_matern = skg.Variogram(X, Y)
V_matern.plot()
    


#%% Generate data
N = 1000
rho = 0.2
spatial_var = 1
noise_var = 0.1
nu = 0.5
X, Y = gen_matern(N, rho, spatial_var, noise_var, nu)
coords_matern = X[:, 1:3]
plt.subplot(1,2,1)
plt.scatter(X[:,1], X[:,2], s = 20, c = Y)

plt.subplot(1,2,2)
X1, Y1 = gen_expo(N, rho, spatial_var, noise_var)
coords_expo = X1[:, 1:3]
plt.scatter(X1[:,1], X1[:,2], s = 20, c = Y1)


#%% Variogram analysis
V_matern = skg.Variogram(coords_matern, Y)
V_matern.plot()


V_expo = skg.Variogram(coords_expo, Y1)
V_expo.plot()


#%% Verfiy equation 8
# Generate data on a grid
spatial_var = 100
rho = 0.2
nu = 1
kappa = np.sqrt(8 * nu) / rho

n_coord = 40
N = n_coord**2

mini, maxi = 0, 1
axis = np.linspace(mini, maxi, n_coord)
X, Y = np.meshgrid(axis, axis)
coords = np.stack([X, Y]).transpose(1,2,0).reshape(-1,2)

# Exponential Correlation
distance = distance_matrix(coords.reshape(-1, 2), coords.reshape(-1, 2))
corr = Matern_Cor(nu, rho, distance)
# Cholesky decomposition and generate correlated data
L = np.linalg.cholesky(spatial_var*corr)
z = np.random.normal(0, 1, N)
Y = np.dot(L, z)
plt.scatter(coords[:,0], coords[:,1], s = 20, c = Y)

# Take difference
Y2 = Y.reshape((40, 40))
Y2_x = np.zeros((40, 39))
for i in range(39):
    Y2_x[:, i] = Y2[:, i+1] - Y2[:, i]
Y2_xx = np.zeros((40, 38))
for i in range(38):
    Y2_xx[:, i] = Y2_x[:, i+1] - Y2_x[:, i]
    
#Y2_y = np.zeros((39, 40))
#for i in range(39):
#    Y2_y[i, :] = Y2[i+1, :] - Y2[i, :]
#Y2_yy = np.zeros((38, 40))
#for i in range(38):
#    Y2_yy[:, i] = Y2_y[i+1, :] - Y2_y[i, :]
    
Y2_y = np.zeros((39, 38))
for i in range(39):
    Y2_y[i, :] = Y2_xx[i+1, :] - Y2_xx[i, :]
Y2_yy = np.zeros((38, 38))
for i in range(38):
    Y2_yy[:, i] = Y2_y[i+1, :] - Y2_y[i, :]

delta = np.zeros((40, 40))
delta[1:39, 1:39] = Y2_yy
    
w = kappa**2 * Y2 - delta
w = w.reshape(-1)

delta_flat = Y2_yy.reshape(-1)

plt.hist(delta_flat, bins=30, color='skyblue', edgecolor='black')
#plt.hist(w, bins=30, color='skyblue', edgecolor='black')

delta_mean = np.mean(delta_flat)
delta_var = np.var(delta_flat)

w_mean = np.mean(w)
w_var = np.var(w)

print(f'delta mean: {delta_mean}; delta var: {delta_var}')
print(f'w mean: {w_mean}; w var: {w_var}')

#%%

Y2_x = np.zeros((40, 39))
for i in range(39):
    Y2_x[:, i] = Y2[:, i+1] - Y2[:, i]
Y2_xx = np.zeros((40, 38))
for i in range(38):
    Y2_xx[:, i] = Y2_x[:, i+1] - Y2_x[:, i]


Y2_y = np.zeros((39, 40))
for i in range(39):
    Y2_y[i, :] = Y2[i+1, :] - Y2[i, :]
Y2_yy = np.zeros((38, 40))
for i in range(38):
    Y2_yy[i, :] = Y2_y[i+1, :] - Y2_y[i, :]
    
delta_x = delta_y = np.zeros((40, 40))
delta_x[:, 0:38] = Y2_xx
delta_y[0:38, :] = Y2_yy

delta = delta_x + delta_y

w = kappa**2 * Y2 - delta
w = w.reshape(-1)


#%% Log Normal process
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
N = 1000
rho = 0.2
noise_var = 0.1
nu = 1
spatial_var = 1
X, Y = gen_lognormal(N, rho, spatial_var, noise_var, nu)
plt.scatter(X[:, 1], X[:, 2], s = 20, c = Y)
plt.title("Log-Normal Process")


# %%A mixture of Gaussian field with different smoothness
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

n = 1000
spatial_var = 1
noise_var = 0.1
nu1 = 2.5
rho1 = 0.3
nu2 = 0.5
rho2 = 0.1

X, Y = gen_mixture(n, spatial_var, noise_var, nu1, nu2, rho1, rho2)
plt.scatter(X[:, 1], X[:, 2], s = 20, c = Y)

s = X[:, 1:3]
V_matern = skg.Variogram(s, Y)
V_matern.plot()












