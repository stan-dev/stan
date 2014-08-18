transformed data {
  real y;
  y <- get_lp();
}
parameters {
  real z;
}
model {
  z ~ normal(0,1);
}
