# How the weight parameter might affect the performance of PINN
# Simulate a stationary process and apply NN
#%% Packages
import sys
wk_dir = "/Users/hongjianyang/PINN_SPDE/"
sys.path.append(wk_dir)
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from scipy.spatial import distance_matrix
import torch.nn as nn
import torch.optim as optim
from spde import *
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')
    

#%% Read data
data = pd.read_csv("non_stat_200.csv", index_col=None)
data = np.array(data.drop(data.columns[0], axis = 1))
X = data[:, 0:3]
Y = data[:, 3]

X_train, X_test, y_train, y_test = random_split(X, Y)


model = model_train_pinn(X_train, y_train, layers, lr, epochs, alpha)

#%%
temp = MSE / iters
plt.plot(alphas, temp)