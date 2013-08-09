data {
  int<lower=0> N;
  vector[N] y;
} 
parameters {
  real mu_theta;
  real<lower=0,upper=100> sigma_theta;
  real<lower=0,upper=100> sigma_y;
}
transformed parameters {
  vector[N] e_theta;

  e_theta <- theta - mu_theta;
} 
model {
  mu_theta ~ normal(0, 1);
  sigma_theta ~ uniform(0, 100);
  sigma_y ~ uniform(0, 100);

  theta ~ normal(100 * mu_theta, sigma_theta);
  y ~ normal(theta, sigma_y);
}
