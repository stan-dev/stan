parameters {
  real<lower=1,scale=-31> a;
  real y;
}

model {
  y ~ normal(0,1);
}
