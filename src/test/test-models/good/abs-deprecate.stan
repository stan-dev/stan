transformed data {
  real mu;
  mu = fabs(-1.2);
}
parameters {
  real y;
}
model {
  y ~ normal(mu, 1);
}

