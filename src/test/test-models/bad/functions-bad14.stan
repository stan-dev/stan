functions {
  real abc(real x) {
    return x;
  }
  int abc(real x) {  // illegal redeclaration with same args
    return 1;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
