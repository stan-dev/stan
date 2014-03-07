parameters {
  real y;
}
model {
  (y * y) ~ normal(0,1);
}
