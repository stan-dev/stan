transformed data {
  real x;
  x <- normal_rng(0,1);
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
