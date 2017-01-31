transformed data {
  int td_a1[2] = 1;   // array <- scalar - bad
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
