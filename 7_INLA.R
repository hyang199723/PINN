setwd("/Users/hongjianyang/PINN/")
library(ggplot2)
library(viridis)
library(spNNGP)
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

iters = 50
MSE = rep(0, iters)
for (i in 1:iters) {
  # Matern simulation
  print(i)
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
  
  g <- ggplot(full, aes(x = Lon, y = Lat, color = Y)) +
    geom_point() +
    theme_minimal() +
    labs(title = "Training data") + 
    scale_color_viridis(option = "D")
  #g
  # Fitting using INLA package
  library(INLA)
  library(fields)
  
  train_row = sample(1:N, 800)
  train = full[train_row, ]
  test = full[-train_row, ]
  s = 1
  rownames(train) = NULL
  rownames(test) = NULL
  train_loc = as.matrix(train[, 1:2])
  test_loc = as.matrix(test[, 1:2])
  
  # INLA 
  mesh <- inla.mesh.2d(loc = train_loc, max.edge = s * 1.1)  # Define the mesh
  spde_model <- inla.spde2.matern(mesh = mesh, alpha = 2)
  s.index <- inla.spde.make.index(name = "spatial.field", 
                                  n.spde = spde_model$n.spde) # index corresponding to data sites
  A_train <- inla.spde.make.A(mesh = mesh, loc = train_loc) # create projection matrix
  A_test <- inla.spde.make.A(mesh = mesh, loc = test_loc) # create projection matrix
  stack_train <- inla.stack(data  = list(y = train$y),
                            A = list(A_train, 1),
                            effects = list(c(s.index, list(Intercept = 1)),
                                           list(Lon = train$Lon,
                                                Lat = train$Lat)),
                            tag = "train.data")
  
  plot(mesh)
  #formula <- y ~ 1 + f(train[, 0:2], model = spde_model, mesh = mesh)  # Replace 'y' and 'location' with your variables
  formula <- y ~ -1 + Intercept + Lon + Lat + f(spatial.field, model = spde_model)
  
  #fit_1 <- inla(formula, data = inla.stack.data(stack_train), family = "gaussian",
  #             control.predictor = list(A = inla.stack.A(stack_train), compute = TRUE))
  
  # Prediction
  stack_test <- inla.stack(data = list(y = NA), 
                           A = list(A_test, 1), tag = "test", 
                           effects = list(c(s.index, list(Intercept = 1)),
                                          list(Lon = test$Lon,
                                               Lat = test$Lat)))
  
  join.stack <- inla.stack(stack_train, stack_test)
  fit_2 <- inla(formula, data = inla.stack.data(join.stack), family = "gaussian",
                control.predictor = list(A = inla.stack.A(join.stack), compute = TRUE))
  
  # Get testing data
  index.test <- inla.stack.index(join.stack, "test")$data
  yhat = fit_2$summary.fitted.values[index.test, "mean"] # Check mean
  
  MSE[i] = mean((test$y - yhat)^2)
}
# nnGP (-)
# GP-GP
# Basic Kriging Result


plot(x = test$y, y = yhat)
