parameters {
  real<lower=0> y;
}
transformed parameters {
  print("target = ", target());
  print("get_lp = ", target());
}
model {
  print("target = ", target());
  print("get_lp = ", target());
  y ~ normal(0, 1);
}

