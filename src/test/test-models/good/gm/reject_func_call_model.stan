functions {
  void foo(real x) {
    reject("user-specified rejection");
  }
}
parameters {
  real y;
}
model {
  foo(y);
  y ~ normal(0,1);
}
