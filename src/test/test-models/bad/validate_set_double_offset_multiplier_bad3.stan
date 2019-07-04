parameters {
  real<lower=1,multiplier=-31> a;
  real y;
}

model {
  y ~ normal(0,1);
}
