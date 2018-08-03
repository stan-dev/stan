parameters {
  real y;
}
model {
  real<lower=0,upper=1> a;
  y ~ normal(0, 1);
}
