/*
 * Double DEF with Poisson observations, normal-(log-normal) hierarchy
 */
data {
  int<lower=0> N;      // number of data points
  int<lower=0> I;      // number of count types
  int<lower=0> x[N,I]; // x[n,i]: count of type i for observation n

  int<lower=0> K2; // number of latent variables in layer 2
  int<lower=0> K1; // number of latent variables in layer 1

  // Fixed hyperparameters of latent variable DEF
  real          mu_W1;
  real<lower=0> std_W1;
  real          mu_z2;
  real<lower=0> std_z2;
  real<lower=0> std_z1;

  // Fixed hyperparameters of weight DEF
  real          mu_W_W1;
  real<lower=0> std_W_W1;
  real          mu_W_z1;
  real<lower=0> std_W_z1;
  real<lower=0> std_W0;
}

parameters {
  // Parameters of latent variable DEF
  matrix[K2, K1]         W1;
  matrix[N, K2]          z2;
  matrix<lower=0>[N, K1] z1;

  // Parameters of weight DEF
  matrix[K2, I]         W_W1;
  matrix[I, K2]         W_z1;
  matrix<lower=0>[K1, I]  W0;
}

model {
  // Log-normal DEF for latent variables
  // Weight priors
  for (k in 1:K2)
    W1[k] ~ normal(mu_W1, std_W1);
  // Latent variable
  for (n in 1:N) {
    z2[n] ~ normal(mu_z2, std_z2);
    for (k in 1:K1)
      z1[n, k] ~ lognormal( tanh(z2[n] * col(W1, k)) , std_z1 );
  }

  // Log-normal DEF for weights
  // Weight priors
  for (k in 1:K2)
    W_W1[k] ~ normal(mu_W_W1, std_W_W1);
  // Latent variable
  for (i in 1:I) {
    W_z1[i] ~ normal(mu_W_z1, std_W_z1);
    for (k in 1:K1)
      W0[i, k] ~ lognormal( tanh(W_z1[i] * col(W_W1, k)) , std_W0 );
  }

  // Observation model
  for (n in 1:N)
    x[n] ~ poisson(z1[n] * W0);
}
