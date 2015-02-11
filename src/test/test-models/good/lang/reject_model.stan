data {
}
parameters {
  real<lower=-10, upper=10> y[2];
}
transformed parameters {
  real<lower=0> z[2];
  z[1] <- exp(y[1]);
  z[2] <- exp(y[2]) * exp(y[1]);
}
model {
  y ~ normal(0,1);
  reject("user-specified rejection");
}

