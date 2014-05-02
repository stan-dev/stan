functions {
  real foo^3(real x) {
    return 2.0 * x;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
