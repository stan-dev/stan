data {
  vector[7] b0;
}
transformed data {
  vector[8] td_b2 = b0;  // can't check dimension sizes at compile time
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
