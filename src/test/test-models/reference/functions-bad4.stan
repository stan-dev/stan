functions {
  real bar(real x) {
    return 2 * x;
  }
}
parameters {
  real y;
}
model {
  bar(y);  // illegal use of non-void function as statement
  y ~ normal(0,1);
}
