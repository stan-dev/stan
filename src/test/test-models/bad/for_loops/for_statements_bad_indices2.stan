functions {
  int foo(int a) {
    while (1) {
      int vs[2,3];
      for (v in vs) v = 3;
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

