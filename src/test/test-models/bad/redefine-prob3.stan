functions {
  real foo_log(int y) {
    return -y^2;
  }
  real foo_lpmf(int y) {
    return y^2 / 2;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0, 1);
}
