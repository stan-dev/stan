data {
  real a1;
}
transformed data {
  int td_a1 = 4.1;    // int_d <- real val - bad
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
