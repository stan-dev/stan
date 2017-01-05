data {
  real d_r;
}
parameters {
  real y;
  real z = d_r;
}
model {
  y ~ normal(0,1);
}
