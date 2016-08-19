transformed data {
  real x;
  x = multiply_log(x, x);  // should raise deprecation warning
  x = binomial_coefficient_log(x, x);  // ditto
  x = lmultiply(x, x);  // new function is OK
  x = lchoose(x, x);  // new function is OK
}
parameters {
  real y;
}
model {
  y ~ normal(0, 1);
}
