parameters {
  real y;
}
model {
  real z;
  z = normal_rng(0, 1);

  y ~ normal(0, 1);
}
