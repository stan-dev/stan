parameters {
  real y;
}
model {
  int x;
  //  x = x && x;
  x = x ? 1 : 2;
  y ~ normal(0,1);
}
