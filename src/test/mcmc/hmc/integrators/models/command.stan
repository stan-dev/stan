data {
  real y;
}
parameters {
  real mu;
}
model {
  y ~ normal(mu,1);
}
