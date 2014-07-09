functions {
  void foo(real x) {
    print("x=",x);
    raise_exception("user-specified exception");
  }
}
transformed data {
  real<lower=0> x;
  foo(x);
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
