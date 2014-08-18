functions {
  void foo_bar() {
    raise_exception("user-specified exception");
  }
}
parameters {
  real y;
}
transformed parameters {
  real x;
  foo_bar();
}
model {
  y ~ normal(0,1);
}
