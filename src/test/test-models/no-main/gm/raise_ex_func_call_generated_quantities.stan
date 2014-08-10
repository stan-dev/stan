functions {
  void foo(real x) {
    raise_exception("user-specified exception");
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
generated quantities {
  foo(y);
}
