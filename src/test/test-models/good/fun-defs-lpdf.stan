functions {
  real foo_lpdf(real a, real b) {
    return a / b;
  }
}
parameters {
  real y;
}
model {
  y ~ foo(3);
}
