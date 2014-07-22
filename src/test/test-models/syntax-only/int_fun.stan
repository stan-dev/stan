functions {
  int foo(int x) {
    return x + 1;
  }
}
transformed data {
  int x;
  x <- foo(2);
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
