parameters {
  real y;
}
model {
  int y;
  vector[3] x;
  if (y)
    y ~ normal(0,1);
  else if (x)               // ERROR HERE
    y ~ normal(0,1);
}
