transformed data {
  real<lower=0> x;
  raise_exception("user-specified exception");
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
generated quantities {
  print("generating quantities");
}

