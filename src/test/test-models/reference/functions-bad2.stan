functions {
  real foo(real x);
  // error not defining foo
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
