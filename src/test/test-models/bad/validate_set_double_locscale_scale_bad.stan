parameters {
  real<scale=-31> a;
  real y;
}

model {
  y ~ normal(0,1);
}
