transformed data {
  vector[5] mu[17];
  mu[1][2] <- 118.22;
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
