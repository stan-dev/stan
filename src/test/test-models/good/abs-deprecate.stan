transformed data {
  real mu;
  mu = abs(-1.2);
}
parameters {
  real y;
}
model {
  y ~ normal(mu, 1);
}
