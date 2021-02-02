data {
  real L;
  real U;
}
parameters {
  real<lower=L, upper=U> infty;
}
model {
  infty ~ normal(0, 1);
}

