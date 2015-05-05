parameters {
  real y;
}
transformed parameters {
  real<lower=0> x;
  reject("user-specified rejection");
}
model {
  y ~ normal(0,1);
}
generated quantities {
  print("generating quantities");
}
