functions {
  real foo_log(real y) {
    return -y^2;
  }
  real foo_lpmf(real y) {
    return y^2 / 2;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0, 1);
}
