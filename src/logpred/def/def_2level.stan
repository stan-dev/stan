data {
  int<lower=0> N;      // number of data points
  int<lower=0> I;      // number of count types
  int<lower=0> x[N,I]; // x[n,i]: count of type i for observation n

  int<lower=1> K_2; // number of latent variables in layer 2
  int<lower=1> K_1; // number of latent variables in layer 1

  // Fixed hyperparameters
  real<lower=0> alpha_w1; // alpha for W_1
  real<lower=0> alpha_w0; // alpha for W_0

  real<lower=0> beta_w1;  // beta for W_1
  real<lower=0> beta_w0;  // beta for W_0

  real<lower=0> alpha_z2; // alpha for z_2
  real<lower=0> beta_z2;  // beta  for z_2

  real<lower=0> alpha_z1; // alpha for z_1
}

parameters {
  // Weights
  matrix<lower=0>[K_2, K_1] W_1;
  matrix<lower=0>[K_1, I]   W_0;

  // Latent variables
  matrix<lower=0>[N, K_2] z_2;
  matrix<lower=0>[N, K_1] z_1;
}

model {
  // Weight priors
  for (k in 1:K_2)
    W_1[k] ~ gamma(alpha_w1, beta_w1);
  for (k in 1:K_1)
    W_0[k] ~ gamma(alpha_w0, beta_w0);

  // Deep exponential family
  for (n in 1:N) {
    z_2[n] ~ gamma(alpha_z2, beta_z2);
    for (k in 1:K_1)
      z_1[n, k] ~ gamma(alpha_z1, alpha_z1/(z_2[n] * col(W_1, k)));
  }

  // Observation model
  for (n in 1:N)
    x[n] ~ poisson(z_1[n] * W_0);
}
