#
# Simulated data
# Sparse Gamma DEF (Ranganath et al., 2015)
#
library(rstan)

# Data
N <- 500
I <- 5
x <- matrix(NA, nrow=N, ncol=I)

K_1 <- 15
K_2 <- 30
K_3 <- 100
alpha_0w0 <- alpha_0w1 <- alpha_0w2 <- 0.1
beta_0w0 <- beta_0w1 <- beta_0w2 <- 0.3
alpha_0z3 <- alpha_0z2 <- alpha_0z1 <- 0.3
beta_0z3 <- 0.3

# Parameters
# Above settings lead to
# N*K_1 + N*K_2 + N*K_3 + K_1*I + K_2*K_1 + K_3*K_2 = 76,025 parameters
# NYT data: N=166,000 documents and I=8,000 terms; 24,193,450 parameters
z_1 <- matrix(NA, nrow=N, ncol=K_1)
z_2 <- matrix(NA, nrow=N, ncol=K_2)
z_3 <- matrix(NA, nrow=N, ncol=K_3)
W_0 <- matrix(NA, nrow=K_1, ncol=I)
W_1 <- matrix(NA, nrow=K_2, ncol=K_1)
W_2 <- matrix(NA, nrow=K_3, ncol=K_2)

# Model
set.seed(42)
# Weight priors
for (k in 1:K_1) {
  W_0[k, ] <- rgamma(I, alpha_0w0, beta_0w0)
}
for (k in 1:K_2) {
  W_1[k, ] <- rgamma(K_1, alpha_0w1, beta_0w1)
}
for (k in 1:K_3) {
  W_2[k, ] <- rgamma(K_2, alpha_0w2, beta_0w2)
}
# Deep exponential family
for (n in 1:N) {
  z_3[n, ] <- rgamma(K_3, alpha_0z3, beta_0z3)
  for (k in 1:K_2) {
    z_2[n, k] <- rgamma(1, alpha_0z2, alpha_0z2/(z_3[n, ] %*% W_2[, k]))
  }
  for (k in 1:K_1) {
    z_1[n, k] <- rgamma(1, alpha_0z1, alpha_0z1/(z_2[n, ] %*% W_1[, k]))
  }
}
# Observation model
for (n in 1:N) {
  for (i in 1:I) {
    x[n, i] <- rpois(1, z_1[n, ] %*% W_0[, i])
  }
}

stan_rdump(
  c("N", "I", "x",
  "K_1", "K_2", "K_3",
  "alpha_0w0", "alpha_0w1", "alpha_0w2",
  "beta_0w0", "beta_0w1", "beta_0w2",
  "alpha_0z3", "alpha_0z2", "alpha_0z1",
  "beta_0z3"),
  "sparse_gamma_def.data.R")
