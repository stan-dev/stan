data {
  vector[7] b0;
}
transformed data {
  vector[8] td_b2 = b0;
}
parameters {
  real y;
}
model {
  y ~ normal(0, 1);
}

