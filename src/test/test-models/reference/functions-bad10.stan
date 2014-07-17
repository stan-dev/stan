functions {
  real badrng(real x) {
    return 3.0 + 2.0 * normal_rng(0,1);
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
