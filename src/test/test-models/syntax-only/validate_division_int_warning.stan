transformed data {
  int x;
  x <- 2 / 3;
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
