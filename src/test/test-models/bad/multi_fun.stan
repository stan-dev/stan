functions {
  int foo(int a) {
    return a;
  }
  real foo(real b) {
    return b;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0, 1);
}
