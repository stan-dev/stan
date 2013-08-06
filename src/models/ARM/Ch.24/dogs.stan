data {
  int<lower=0> n_trials;
  int<lower=0> n_dogs;
  matrix[n_dogs,n_trials] y;
}
parameters {
  real<lower=0> sigma_beta;
  vector[3] beta;
  real mu_beta;
}
transformed parameters {
  matrix[n_dogs,n_trials] n_avoid;
  matrix[n_dogs,n_trials] n_shock;
  matrix[n_dogs,n_trials] p;
  
  for (j in 1:n_dogs) {
    n_avoid[j,1] <- 0;
    n_shock[j,1] <- 0;
    for (t in 2:n_trials) {
      n_avoid[j,t] <- n_avoid[j,t-1] + 1 - y[j,t-1];
      n_shock[j,t] <- n_shock[j,t-1] + y[j,t-1];
    }
    for (t in 1:n_trials)
      p[j,t] <- beta[1] + beta[2] * n_avoid[j,t] + beta[3] * n_shock[j,t];
  }
}
model {
  mu_beta ~ normal(0, 100);
  sigma_beta ~ uniform(0, 100);
  beta ~ normal(mu_beta, sigma_beta);

  y ~ bernoulli_logit(p);
}
