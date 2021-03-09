transformed data {
  real x;
  x = lmultiply(x, x);
  x = lchoose(x, x);
  x = lmultiply(x, x);
  x = lchoose(x, x);
}
parameters {
  real y;
}
model {
  y ~ normal(0, 1);
}

