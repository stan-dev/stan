parameters {
  real y;
}
transformed parameters {
  real<lower=0> x;
  raise_exception("user-specified exception");
}
model {
  y ~ normal(0,1);
}
generated quantities {
  print("generating quantities");
}
