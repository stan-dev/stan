functions {
  vector bizbuz_log(vector x) {
    return exp(x);
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
