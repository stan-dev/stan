data {
  real y;
  real<lower=0> sigma_y;
  real mu_0;
  real<lower=0> sigma_0;
}
parameters {
  real theta;
}
model {
  y ~ normal(theta, sigma_y);
  theta ~ normal(mu_0, sigma_0);
}

