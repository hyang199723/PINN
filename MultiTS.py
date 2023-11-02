#%% Packagesimport syswk_dir = "/Users/hongjianyang/PINN_SPDE/"sys.path.append(wk_dir)import torchimport numpy as npimport matplotlib.pyplot as pltimport pandas as pdimport numpy as npimport randomfrom sklearn.metrics import pairwise_distancesfrom sklearn.model_selection import train_test_splitimport torch.nn as nnimport torch.optim as optimfrom spde import *if torch.cuda.is_available():    device = torch.device('cuda')else:    device = torch.device('cpu')    #%% Simulate dataN = 200rho = 3vvv = 2T = 10 # number of time stepsX, Y = gen_multiTS(N, rho, vvv, T)X_train, X_test, y_train, y_test = random_split(X, Y)plt.subplot(2,3,1)plt.scatter(X[:,1], X[:,2], s = 20, c = Y[:, 0])plt.subplot(2,3,2)plt.scatter(X[:,1], X[:,2], s = 20, c = Y[:, 1])plt.subplot(2,3,3)plt.scatter(X[:,1], X[:,2], s = 20, c = Y[:, 2])plt.subplot(2,3,4)plt.scatter(X[:,1], X[:,2], s = 20, c = Y[:, 3])plt.subplot(2,3,5)plt.scatter(X[:,1], X[:,2], s = 20, c = Y[:, 4])plt.colorbar()