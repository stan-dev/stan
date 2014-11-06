functions {
  void foo(real x) {
    print("x=",x);
  } 
}
parameters {
  real y;
}
model {
  foo(y);
  y ~ normal(0,1);
}
