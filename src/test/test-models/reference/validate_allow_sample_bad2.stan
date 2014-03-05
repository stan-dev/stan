parameters {
  real y;
}
transformed parameters {
  real z;
  z ~ normal(0,1);
}
model {
  y ~ normal(0,1);
}
