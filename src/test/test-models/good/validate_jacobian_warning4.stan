parameters {
  real y;
}
model {
  1 + (y * y) ~ normal(0,1);
}
