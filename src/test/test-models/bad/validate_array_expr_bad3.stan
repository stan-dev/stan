parameters {
  real y;
}
model {
  int int_1_a[3];
  int_1_a = { };  // cannot be empty
  y ~ normal(0,1);
}
