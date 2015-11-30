parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  // TODO(carpenter): why can't we get this unconstrained var out?
  // params_r__ ~ normal(0, 1);
  1 ~ normal(mu, sigma);
  2 ~ normal(mu, sigma);
  3 ~ normal(mu, sigma);
}
