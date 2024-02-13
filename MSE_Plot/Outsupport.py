#%% System parameters
import sys
import pandas as pd
import numpy as np
import matplotlib as plt
wk_dir = "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/PINN/"
df = pd.read_csv(wk_dir + "Output/outsupport_MSE.csv")
# %% Remove first column
mse = df.iloc[:, 1:df.shape[1]]

# %%
mean = np.mean(mse, axis = 0)
sd = np.std(mse, axis = 0)
# %% Plot error plot
plt.figure(figsize=(10, 6))
plt.errorbar(x=mse.columns, y=mean, yerr=sd, fmt='o-', capsize=5, label="OutSupport")
plt.legend()
plt.title("MSE comparison over different alpha, testing data out of support")

# %%
