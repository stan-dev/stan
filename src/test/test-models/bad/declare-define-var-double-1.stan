data {
  real a1[2];
}
transformed data {
  real td_a1 = a1;   // scalar <- array - bad
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}

