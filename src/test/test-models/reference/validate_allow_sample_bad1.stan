transformed data {
  real z;
  z ~ normal(0,1);
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
