functions {
  real badassign(real x) {
    x <- 5;
    return x;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
