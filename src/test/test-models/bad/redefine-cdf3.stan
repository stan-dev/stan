functions {
  real foo_lcdf(int n, real x) {
    return -x^2;
  }
  real foo_cdf_log(int n, real x) {
    return -x^2;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0, 1);
}
