functions {
  real foo_log(real y, real theta) { return 1; }
  real foo_cdf_log(real y, real theta) { return 1; }
  real foo_ccdf_log(real y, real theta) { return 1; }
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
