data {
  int<lower=0> N;
  vector[N] y;
} 
parameters {
  real<lower=0> sigma_y;
  real<lower=0> sigma_theta;
  real mu_theta;
}
transformed parameters {
  vector[N] e_theta;

  e_theta <- theta - mu_theta;
} 
model {
  mu_theta ~ normal(0, .0001);
  sigma_theta ~ uniform(0, 100);
  sigma_y ~ uniform(0, 100);

  theta ~ normal(mu_theta, sigma_theta);
  y ~ normal(theta, sigma_y);
}
