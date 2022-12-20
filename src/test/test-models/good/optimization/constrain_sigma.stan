parameters {
  real<lower=0> sigma;
}
model {
  sigma ~ normal(3, 1);
}
