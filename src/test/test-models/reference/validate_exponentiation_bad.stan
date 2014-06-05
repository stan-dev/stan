data {
  int i;
  int j;
}
transformed data {
  int z;
  z <- i ^ j;  // int, int
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
