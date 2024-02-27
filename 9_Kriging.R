# Kriging with true parameters and estimated parameters with GPGP
setwd("/Users/hongjianyang/PINN/")
library(ggplot2)
library(viridis)
library(GpGp)
library(maps)
library(geoR)
dat = read.csv("Data/matern_02_1_5.csv", header = FALSE)
# Recover the original 3D dataframe
iters = 100
full_3D = array(0, dim = c(2000, 3, iters))
for (i in 1:iters) {
  slice = dat[, i]
  lon = slice[seq(1, length(slice), by = 3)]
  lat = slice[seq(2, length(slice), by = 3)]
  y = slice[seq(3, length(slice), by = 3)]
  full_3D[,1,i] = lon
  full_3D[,2,i] = lat
  full_3D[,3,i] = y
}
Ns = c(400, 600, 800, 1000, 1500)
nnn = length(Ns)
MSE = rep(0, nnn)
MSE_est = rep(0, nnn)

for (j in 1:nnn) {
  n = Ns[j]
  for (i in 1:1) {
    full = full_3D[1:n,,i]
    # Train test split
    train_row = sample(1:n, n * 0.8)
    train = full[train_row, ]
    test = full[-train_row, ]
    rownames(train) = NULL
    rownames(test) = NULL
    X = rep(1, dim(train)[1])
    X0 = rep(1, dim(test)[1])
    s = as.matrix(train[, 1:2])
    s0 = as.matrix(test[, 1:2])
    pred <- krige.conv(data=train[,3],coords=s, 
                       locations=s0,
                       krige=krige.control(cov.model="matern",
                                           cov.pars=c(5, 0.2), # spatial var and range
                                           kappa = 1,
                                           nugget=5))
    yhat = pred$predict
    mse = mean((yhat - test[,3])^2)
    
    # Use GPGP to estimate spatial cov
    fit <- fit_model(train[,3], s, rep(1, n * 0.8), "matern_isotropic",  m_seq = c(10, 30))
    params = fit$covparms
    vvv = params[1]
    range = params[2]
    smooth = params[3]
    nugget = params[4]
    pred_est <- krige.conv(data=train[,3],coords=s, # Describe training data
                           locations=s0,    # Describe prediction sites
                           krige=krige.control(cov.model="matern",
                                               cov.pars=c(vvv, range), # spatial var and range
                                               kappa = smooth,
                                               nugget=vvv * nugget))
    yhat_est = pred_est$predict
    mse_est = mean((yhat_est - test[,3])^2)
    MSE[j] = MSE[j] + mse
    MSE_est[j] = MSE_est[j] + mse_est
  }
}
  
  
