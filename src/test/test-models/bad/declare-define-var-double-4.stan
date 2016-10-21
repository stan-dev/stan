data {
  real a1[2];
}
parameters {
  real y;
  real a2 = 1.0; // cannot assign in parameters block
}
model {
  y ~ normal(0,1);
}

