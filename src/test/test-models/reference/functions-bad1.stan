functions {
  real foo(real x);
  real foo(real x) {
    return x;
  }
  real foo(real x); // error redeclaring function
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
