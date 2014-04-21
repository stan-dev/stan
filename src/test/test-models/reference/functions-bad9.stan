functions {
  real foobar_rng(real x) {
    return 3.0 + 2.0 * normal_rng(0,1);
  }
}
parameters {
  real y;
}
transformed parameters {
  real z;
  z <- foobar_rng(1.3);
}
model {
  y ~ normal(0,1);
}
