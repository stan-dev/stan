parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  params_r__ ~ normal(0, 1);
  1 ~ normal(mu, sigma);
  2 ~ normal(mu, sigma);
  3 ~ normal(mu, sigma);
}
