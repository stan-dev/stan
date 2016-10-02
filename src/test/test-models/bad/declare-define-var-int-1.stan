data {
  real a1;
}
transformed data {
  int td_a1 = a1;   // int_d <- real_d - bad
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
