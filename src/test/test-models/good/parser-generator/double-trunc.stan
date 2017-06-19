parameters {
  real y;
}
model {
  y ~ normal(0, 1) T[-1, 1];
}
