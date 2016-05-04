parameters {
  real y;
}
model {
  int x;
  x = 0;
  x = x && x;
  x = x == x || x;
  //  x = x ? x : x;
  y ~ normal(0,1);
}
