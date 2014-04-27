functions {
  real my_fun(real x) {
    return 2 * x;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
