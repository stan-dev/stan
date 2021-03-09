functions {
  real foo_lpdf(real y, real theta) {
    return 1;
  }
  real foo_lcdf(real y, real theta) {
    return 1;
  }
  real foo_lccdf(real y, real theta) {
    return 1;
  }
}
data {
  real y;
}
parameters {
  real theta;
  real L;
  real U;
}
model {
  y ~ foo(theta) T[L, ];
  y ~ foo(theta) T[ , U];
  y ~ foo(theta) T[L, U];
}

