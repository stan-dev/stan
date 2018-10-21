parameters {
  vector[3] vvv;
  real<scale=vvv> a;
  real y;
}

model {
  y ~ normal(0,1);
}
