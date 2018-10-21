parameters {
  vector[3] vvv;
  real<location=vvv> a;
  real y;
}

model {
  y ~ normal(0,1);
}
