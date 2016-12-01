data {
  vector[7] b0;
  row_vector[7] c0;
}
transformed data {
  vector[7] td_b1 = c0;  // base type mismatch
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
