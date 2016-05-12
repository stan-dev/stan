transformed data {
  print("target=", target());
}
parameters {
  real<lower=0> y;
}
model {
  y ~ normal(0, 1);
}
