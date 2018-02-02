parameters {
  real<lower=-10, upper=10> y[2];
}
model {
  y ~ normal(0,1);
}
generated quantities {
  print("no QoIs");
}
