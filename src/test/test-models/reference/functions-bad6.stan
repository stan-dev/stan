functions {
  real foo_lp(real x) {
    return 2.0 * x;
  }
}
transformed data {
  real z;
  z <- foo_lp(1.3);
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
