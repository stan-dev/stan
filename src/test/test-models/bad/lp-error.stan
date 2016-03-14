parameters {
  real<lower=0> y;
}
model {
  lp__ = lp__ - y^2 / 2;
}
