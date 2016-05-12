data {
  real target;
}
parameters {
  real y;
}
model {
  y ~ normal(0, 1);
}
