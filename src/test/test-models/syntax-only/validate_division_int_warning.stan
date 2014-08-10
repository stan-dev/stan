transformed data {
  real u;
  int j;
  int k;
  j <- 2;
  k <- 3;
  u <- j / k;
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
