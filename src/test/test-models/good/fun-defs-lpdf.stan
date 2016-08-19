functions {
  real bar_baz_lpdf(real a, real b) {
    return a / b;
  }
}
parameters {
  real y;
}
model {
  y ~ bar_baz(3.2);
}
