functions {
  int foo(int a) {
    int vs[2];
    int y;
    for (v in vs) y = 3;
    return 0;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
