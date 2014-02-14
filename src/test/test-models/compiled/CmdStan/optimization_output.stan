parameters {
  real y[2,2];
}
model {
  y[1,1] ~ normal(1,1);
  y[2,1] ~ normal(100,1);
  y[1,2] ~ normal(10000,1);
  y[2,2] ~ normal(1000000,1);
}
