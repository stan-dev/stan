// Fit a Gaussian process's hyperparameters
// for squared exponential prior

data {
  int<lower=1> N;
  vector[N] x;
  vector[N] y;
}
transformed data {
  vector[N] mu;
  for (i in 1:N) 
    mu[i] <- 0;
}
parameters {
  real<lower=0> eta_sq;
  real<lower=0> rho_sq;
  real<lower=0> sigma_sq;
}
model {
  matrix[N,N] Sigma;
  for (i in 1:N)
    for (j in i:N)
      Sigma[i,j] <- eta_sq * exp(-rho_sq * pow(x[i] - x[j],2))
                    + if_else(i==j, sigma_sq, 0.0);
  for (i in 1:N)
    for (j in (i+1):N)
      Sigma[j,i] <- Sigma[i,j];

  eta_sq ~ cauchy(0,5);
  rho_sq ~ cauchy(0,5);
  sigma_sq ~ cauchy(0,5);

  y ~ multi_normal(mu,Sigma);
}
