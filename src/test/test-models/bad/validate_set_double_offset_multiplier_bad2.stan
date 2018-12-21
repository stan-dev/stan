parameters {
  vector[3] vvv;
  real<multiplier=vvv> a;
  real y;
}

model {
  y ~ normal(0,1);
}
