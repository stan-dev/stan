parameters {
  real<lower=0> y;
}
model {
  print("target=", target());
  y ~ normal(0, 1);
}

