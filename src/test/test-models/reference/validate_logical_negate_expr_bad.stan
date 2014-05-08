transformed data {
  vector[3] v;
  v <- !v;
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
