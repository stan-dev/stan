transformed data {
  vector v;
  v <- !v;
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
