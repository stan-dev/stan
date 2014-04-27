functions {
  void baz(real x) {
    return;
  }
}
parameters {
  real y;
}
model {
  real z;
  z <- baz(y); // illegal use of void function as expression
  y ~ normal(0,1);
}
