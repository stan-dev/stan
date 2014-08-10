functions {
  void foo(real x) {
    raise_exception("user-specified exception");
  }
}
parameters {
  real y;
}
model {
  foo(y);
  y ~ normal(0,1);
}
