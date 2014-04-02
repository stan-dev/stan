functions {
  real foo_lp(real x) {
    return 2.0 * x;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
generated quantities {
  real z;
  z <- foo_lp(1.3);
}
