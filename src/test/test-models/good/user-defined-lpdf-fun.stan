functions {
  real bar_lpmf(int y, real z) {
    return 1.0;
  }
  real bar_lcdf(int y, real z) {
    return 1.0;
  }
  real bar_lccdf(int y, real z) {
    return 1.0;
  }

  real foo_lpdf(real y, real sigma) {
    return 1.0;
  }
  real foo_lcdf(real y, real sigma) {
    return 1.0;
  }
  real foo_lccdf(real y, real sigma) {
    return 1.0;
  }
}
parameters {
  real y;
}
model {
  target += foo_lpdf(y | 7.0);
  target += foo_lcdf(y | 7.0);
  target += foo_lccdf(y | 7.0);

  target += normal_lpdf(y | 7.0, 1.0);
  target += normal_lcdf(y | 7.0, 1.0);
  target += normal_lccdf(y | 7.0, 1.0);

  target += bar_lpmf(2 | 7.0);
  target += bar_lcdf(2 | 7.0);
  target += bar_lccdf(2 | 7.0);

  target += poisson_lpmf(2 | 7.0);
  target += poisson_lcdf(2 | 7.0);
  target += poisson_lccdf(2 | 7.0);

}
