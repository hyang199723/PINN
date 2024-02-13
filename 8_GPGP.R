# GPGP
setwd("/Users/hongjianyang/PINN/")
library(ggplot2)
library(viridis)
library(GpGp)
gen_matern <- function(N, rho, spatial_var, noise_var, nu) {
  set.seed(123)
  length <- 1
  coords <- matrix(runif(N * 2, 0, length), ncol = 2)
  
  # Creating design matrix X
  X <- cbind(rep(1, N), coords)
  
  # Exponential Correlation
  distance <- as.matrix(dist(coords))
  corr <- Matern(distance, nu = nu, rho = rho)
  
  # Cholesky decomposition and generate correlated data
  L <- chol(spatial_var * corr)
  z <- rnorm(N)
  Y <- L %*% z + rnorm(N, sd = sqrt(noise_var))
  
  return(list(X = X, Y = Y))
}

# Matern function definition
Matern <- function(distance, nu, rho) {
  kappa <- sqrt(8 * nu) / rho
  corr <- 2 ^ (1 - nu) / gamma(nu) * (kappa * distance) ^ nu * besselK(kappa * distance, nu)
  corr[distance == 0] <- 1
  return(corr)
}


#MSE = rep(0, iters)

N = 1000
P = 2
noise_var = 0.1
rho = 0.2
nu = 1
kappa = (8 * nu)**(0.5) / rho
spatial_var = 1
out = gen_matern(N, rho, spatial_var, noise_var, nu)
X = out$X
Y = out$Y

full = data.frame(cbind(X[, 2:3], Y))
colnames(full) <- c("Lon", "Lat", "y")
train_row = sample(1:N, 800)
train = full[train_row, ]
test = full[-train_row, ]

rownames(train) = NULL
rownames(test) = NULL
train_loc = as.matrix(train[, 1:2])
test_loc = as.matrix(test[, 1:2])

train_x = cbind(rep(1, 800), train[, 1:2])
test_x = cbind(rep(1, 200), test[, 1:2])

fit <- fit_model(train$y, train_loc, train_x, "matern_isotropic")

yhat = predictions(fit, test_loc, test_x)
mse = mean((yhat - test$y)^2)

plot(x = test$y, y = yhat)
  
plot(x = test_loc[, 1], y = test_loc[, 2], pch = 19, col = test$y + 2)

plot(x = test_loc[, 1], y = test_loc[, 2], pch = 19, col = yhat + 2)
#################################################################
#################################################################
#################################################################
#################################################################
# Generate data from GpGp and try GpGp
#locs <- as.matrix( expand.grid( (1:50)/50, (1:50)/50 ) )
N = 1000
length <- 1
locs <- matrix(runif(N * 2, 0, length), ncol = 2)
y <- fast_Gp_sim(c(1, 0.2, 1, 0.1), "matern_isotropic",  locs )
# Matern isotropic variance, range, smoothness, nugget

full = data.frame(cbind(locs, y))
colnames(full) <- c("Lon", "Lat", "y")
train_row = sample(1:N, 800)
train = full[train_row, ]
test = full[-train_row, ]


fit <- fit_model(train$y, train_loc, train_x, "matern_isotropic")

yhat = predictions(fit, test_loc, test_x)
mse = mean((yhat - test$y)^2)

write.csv(full, "gpgp_matern.csv")
