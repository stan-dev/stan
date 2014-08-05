parameters {
  real x[3];
  real y[size(x)];
}
model {
  y ~ normal(0,1);
}
