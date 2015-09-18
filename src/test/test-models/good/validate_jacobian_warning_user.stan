parameters {
  vector[1] y;
}
model {
  exp(y[1]) ~ normal(0, 2);
}
