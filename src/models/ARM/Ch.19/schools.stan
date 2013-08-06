data {
  int<lower=0> N;
  vector[N] y;
  vector[N] sigma_y;
} 
parameters {
  real<lower=0> sigma_eta;
  real xi;
  real mu_theta;
  vector[N] eta;
} 
transformed parameters {
  vector[N] theta;
  real<lower=0> sigma_theta;

  theta <- mu_theta + xi * eta;
  sigma_theta <- abs(xi) / sigma_eta;
}
model {
  sigma_eta ~ inv_gamma(1, 1); //prior distribution can be changed to uniform
  mu_theta ~ normal(0, 100);

  eta ~ normal(0, sigma_eta);
  xi ~ normal(0, 5);
  y ~ normal(theta,sigma_y);
}
