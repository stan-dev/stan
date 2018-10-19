parameters {
  real<location=positive_infinity()> a;
  real y;
}

model {
  y ~ normal(0,1);
}
