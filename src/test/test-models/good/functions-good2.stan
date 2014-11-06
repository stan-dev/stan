functions {
  real foo_bar0() {
    return 0.0;
  }
  real foo_bar1(real x) {
    return 1.0;
  }
  real foo_bar2(real x, real y) {
    return 2.0;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
