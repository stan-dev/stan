#
# Simulated data
# DEF with Poisson observations, normal-(log-normal) hierarchy
#
library(rstan)

# Data
N <- 50
I <- 500
x <- matrix(NA, nrow=N, ncol=I)

K2 <- 10
K1 <- 50
std_W0 <- std_W1 <- 1
mu_W0 <- mu_W1 <- 0
std_z1 <- std_z2 <- 1
mu_z2 <- 0

# Parameters
# Above settings lead to
# N*K1 + N*K2  + K1*I + K2*K1 = 28,500 parameters
W1 <- matrix(NA, nrow=K2, ncol=K1)
W0 <- matrix(NA, nrow=K1, ncol=I)
z2 <- matrix(NA, nrow=N, ncol=K2)
z1 <- matrix(NA, nrow=N, ncol=K1)

# Model
set.seed(42)
# Weight priors
for (k in 1:K2)
  W1[k, ] <- rnorm(K1, mu_W1, std_W1)
for (k in 1:K1)
  W0[k, ] <- rlnorm(I, mu_W0, std_W0)

# Deep exponential family
for (n in 1:N) {
  z2[n, ] <- rnorm(K2, mu_z2, std_z2)
  for (k in 1:K1)
    z1[n, k] <- rlnorm(1, tanh(z2[n, ] %*% W1[, k]), std_z1)
}

# Observation model
for (n in 1:N) {
  for (i in 1:I)
    x[n, i] <- rpois(1, z1[n, ] %*% W0[, i])
}

stan_rdump(
  c("N", "I", "x",
  "K1", "K2",
  "std_W0", "std_W1",
  "mu_W0","mu_W1",
  "std_z1", "std_z2"
  "mu_z2"),
  "lognormal_def_2level.data.R")
