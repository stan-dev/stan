parameters {
  real y;
}
model {
  matrix<lower=0,upper=1>[3,4] a[5];
  y ~ normal(0, 1);
}
