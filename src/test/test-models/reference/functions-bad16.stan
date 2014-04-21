functions {
  real square(real x) {
    return x * x;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0.1,1);
}
