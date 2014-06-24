transformed data {
  real x;
  vector[3] v;
  x <- v[1][2];
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
