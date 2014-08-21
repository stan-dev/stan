data {
  real i;
  real j;
}
transformed data {
  real k;
  k <- i % j;  // real, real
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}

