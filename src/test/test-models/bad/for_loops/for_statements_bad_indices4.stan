functions {
  int foo(int a) {
    while (1) {
      matrix[2,3] vs;
      for (v in vs) {
        for (a in v) continue;
      }
    }
    return 0;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
