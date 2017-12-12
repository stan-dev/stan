parameters {
  real<lower=-10, upper=10> y[2];
}
transformed parameters {
  real w;
  real<lower=0> z[2];
  w = y[1];
  z[1] <- exp(y[1]);
  z[2] <- exp(y[2]) * exp(y[1]);
}
model {
  y ~ normal(0,1);
}
generated quantities {
  real xgq = 0.007;
  vector[2] y_rep;
  y_rep[1] = normal_rng(y[1],1);
  y_rep[2] = normal_rng(y[2],1);
}
