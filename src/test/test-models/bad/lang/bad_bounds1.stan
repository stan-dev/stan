data {
  real<lower=1 && 2> x;
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
