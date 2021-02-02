functions {
  int foo(int a) {
    array[2] int vs;
    int y;
    for (v in vs) 
      y = 3;
    return 0;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0, 1);
}

