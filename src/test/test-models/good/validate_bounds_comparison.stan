data {
  real<lower=((1 < 2) ? -1 : -2)> a;
  real<upper=((1 < 2) ? -1 : -2)> b;
  real<lower=(1 && 3), upper=((1 < 2) ? -1 : -2)> c;
}
parameters {
  real y;
}
model {
  y ~ normal(0, 1);
}

