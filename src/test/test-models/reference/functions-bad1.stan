functions {
  real flib(real x);
  real flib(real x) {
    return x;
  }
  real flib(real x); // error redeclaring function
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
