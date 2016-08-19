functions {
  real foo_lpmf(real y, real sigma) {
    return -(y / sigma)^2;
  }
}
parameters {
  real y;
}
model {
  y ~ foo(1.0);
}
