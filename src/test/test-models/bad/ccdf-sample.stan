data {
  vector[3] y;
}
model {
  y ~ weibull_ccdf(1,1);
}
