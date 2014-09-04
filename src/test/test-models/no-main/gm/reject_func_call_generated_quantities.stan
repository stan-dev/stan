functions {
  void foo(real x) {
    reject("user-specified rejection");
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
