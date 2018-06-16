functions {
  int foo(int a) {
    while (1) {
      int xx = 3;
      for (i in xx) continue;
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
