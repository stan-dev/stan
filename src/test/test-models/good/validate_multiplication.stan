transformed data {
  real x;
  x <- 1.3;
  x <- x * 2.7;
  x <- x * x * x;
}
parameters {
  real y;
}
model {
  y ~ normal(x * 3, 1);
}
