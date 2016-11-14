data {
  real a1[2];
}
transformed data {
  vector[2] td_a1 = a1;   // vector <- array - bad
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}

