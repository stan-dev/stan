// TODO the ordering of the layers in the DGP should be reversed
/*
 * Sparse Gamma DEF (Ranganath et al., 2015)
 */
data {
  // Data
  int<lower=0> N;      // number of data points
  int<lower=0> I;      // number of count types
  int<lower=0> x[N,I]; // x[n,i]: count of type i for observation n

  // Fixed hyperparameters
  int<lower=1> K_1; // number of latent variables in layer 1
  int<lower=1> K_2; // number of latent variables in layer 2
  int<lower=1> K_3; // number of latent variables in layer 3
  real<lower=0> alpha_0w0; // alpha for W_0
  real<lower=0> alpha_0w1; // alpha for W_1
  real<lower=0> alpha_0w2; // alpha for W_2
  real<lower=0> beta_0w0;  // beta for W_0
  real<lower=0> beta_0w1;  // beta for W_1
  real<lower=0> beta_0w2;  // beta for W_2
  real<lower=0> alpha_0z3; // alpha for z_3
  real<lower=0> alpha_0z2; // alpha for z_2
  real<lower=0> alpha_0z1; // alpha for z_1
  real<lower=0> beta_0z3;  // beta for z_3
}
parameters {
  // Latent variables
  matrix<lower=0>[N, K_1] z_1;
  matrix<lower=0>[N, K_2] z_2;
  matrix<lower=0>[N, K_3] z_3;

  // Weights
  matrix<lower=0>[K_1, I]   W_0;
  matrix<lower=0>[K_2, K_1] W_1;
  matrix<lower=0>[K_3, K_2] W_2;
}
model {
  // Weight priors
  for (k in 1:K_1)
    W_0[k] ~ gamma(alpha_0w0, beta_0w0);
  for (k in 1:K_2)
    W_1[k] ~ gamma(alpha_0w1, beta_0w1);
  for (k in 1:K_3)
    W_2[k] ~ gamma(alpha_0w2, beta_0w2);

  // TODO am i doing the dot products right, e.g., t(z_3[n,]) %*% W_2
  // Deep exponential family
  for (n in 1:N) {
    z_3[n] ~ gamma(alpha_0z3, beta_0z3);
    for (k in 1:K_2)
      z_2[n, k] ~ gamma(alpha_0z2, alpha_0z2/(z_3[n] * col(W_2, k)));
    for (k in 1:K_1)
      z_1[n, k] ~ gamma(alpha_0z1, alpha_0z1/(z_2[n] * col(W_1, k)));
  }

  // Observation model
  for (n in 1:N) {
    for (i in 1:I)
      x[n, i] ~ poisson(z_1[n] * col(W_0, i));
  }
}
