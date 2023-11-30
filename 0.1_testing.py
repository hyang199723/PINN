# Test function
import sys
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/PINN/"
sys.path.append(wk_dir)

#%% Test KL divergence using scipy
import numpy as np
from scipy.stats import gaussian_kde, norm

# Empirical data points
empirical_data = np.array(np.random.normal(0, 1, 10000))

# Estimating the PDF of the empirical distribution using Kernel Density Estimation
kde = gaussian_kde(empirical_data)

# Define the range over which to evaluate the KDE and theoretical PDF
x = np.linspace(min(empirical_data), max(empirical_data), 1000)

# Evaluate the estimated empirical PDF
empirical_pdf = kde(x)

# Evaluate the theoretical PDF for Normal(0,1) distribution
theoretical_pdf = norm.pdf(x, 0, 1)

# Calculate the KL divergence
# We need to ensure that none of the PDFs have zero values to avoid division by zero in our computation
epsilon = 1e-10  # A small value to ensure numerical stability
empirical_pdf = np.maximum(empirical_pdf, epsilon)
theoretical_pdf = np.maximum(theoretical_pdf, epsilon)

# Calculate the KL divergence using the trapezoidal rule for numerical integration
kl_divergence = np.trapz(empirical_pdf * np.log(empirical_pdf / theoretical_pdf), x)

print(f'The KL divergence is: {kl_divergence}')

#%% Basis function
N = 1000 ##Sample Size
P = 1 ##Covariates
M = 100 ##replicates
X = np.array([np.ones(N)]).T ##Design matrix
#kernel = GPy.kern.Exponential(1,1,0.1) ##Covariance Function
#noise_var = 0.01 ##Nugget variance
# 1000 points evenly spaced over [0,1]
s = np.linspace(0,1,N).reshape(-1,1)

