# GPGP
setwd("/Users/hongjianyang/PINN/")
library(ggplot2)
library(viridis)
library(GpGp)
library(maps)
dat = read.csv("Data/matern_02_1_1.csv", header = FALSE)
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

size = 1000
full = full_3D[1:size,,1]
# Visualize the data
df <- data.frame(long=full[,1],lat=full[,2],Y=full[,3])
ggplot(df, aes(long, lat)) +
  geom_point(aes(colour = Y)) +
  scale_colour_gradientn(colours = viridis(10))
# Train test split
train_row = sample(1:size, size * 0.8)
train = full[train_row, ]
test = full[-train_row, ]

rownames(train) = NULL
rownames(test) = NULL
train_loc = as.matrix(train[, 1:2])
test_loc = as.matrix(test[, 1:2])


fit <- fit_model(train[,3], train_loc, rep(1, size * 0.8), "matern_isotropic",  m_seq = c(10, 30))
# covparams: variance, range, smoothness, nugget
params = fit$covparms
vvv = params[1]
range = params[2]
smooth = params[3]
nugget = params[4]


test_x = rep(1, size * 0.2)
yhat = predictions(fit, test_loc, test_x)
mse = mean((yhat - test[,3])^2)

plot(x = test[, 3], y = yhat)
  
plot(x = test_loc[, 1], y = test_loc[, 2], pch = 19, col = test[,3] + 2)

plot(x = test_loc[, 1], y = test_loc[, 2], pch = 19, col = yhat + 2)
#################################################################
#################################################################
#################################################################
#################################################################
# Generate data from GpGp and try GpGp
#locs <- as.matrix( expand.grid( (1:50)/50, (1:50)/50 ) )
iters = 100
N = 2000
gpgp_data = array(0, dim = c(N * 3, iters))
length <- 1
for (i in 1:iters) {
  slice = rep(0, N*3)
  locs <- matrix(runif(N * 2, 0, length), ncol = 2)
  y <- fast_Gp_sim(c(5, 0.2, 1, 1), "matern_isotropic",  locs, 100)
  lon = locs[,1]
  lat = locs[,2]
  slice[seq(1, length(slice), by = 3)] = lon
  slice[seq(2, length(slice), by = 3)] = lat
  slice[seq(3, length(slice), by = 3)] = y
  gpgp_data[, i] = slice
}
full = cbind(lon, lat, y)
df <- data.frame(long=full[,1],lat=full[,2],Y=full[,3])
ggplot(df, aes(long, lat)) +
  geom_point(aes(colour = Y)) +
  scale_colour_gradientn(colours = viridis(10))
var(y)
write.csv(gpgp_data, "Data/gpgp_02_1_5.csv", row.names = F, col.names = NA)
#dat = read.csv("Data/gpgp_02_1_1.csv", header = T)
# Matern isotropic variance, range, smoothness, nugget
# The nugget value σ2τ2 is added to the diagonal of the covariance matrix. 
# NOTE: the nugget is σ2τ2, not τ2.
fit <- fit_model(y, locs, rep(1, N), "matern_isotropic")
fit$covparms