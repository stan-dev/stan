functions {
  real foo17(voiderama x) {
    return 1.0;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
