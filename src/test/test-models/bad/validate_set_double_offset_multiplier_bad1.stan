parameters {
  vector[3] vvv;
  real<offset=vvv> a;
  real y;
}

model {
  y ~ normal(0,1);
}
