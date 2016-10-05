data {
  matrix[7,8] b0;
}
transformed data {
  vector[7] td_b1 = b0;  // base type mismatch
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
