functions {
  void badlp(real x) {
    x ~ normal(0,1);
    return;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
