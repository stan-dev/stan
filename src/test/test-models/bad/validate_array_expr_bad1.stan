parameters {
  real y;
}
model {
  int int_1_a[5];
  int_1_a = { 1.0, 2.0, 3.0, 4.0 , 5.0 };  // type mismatch
  y ~ normal(0,1);
}
