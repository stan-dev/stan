transformed data {
  real x;
  x <- exp(x);
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
