parameters {
  real y;
}
model {
  int z;
  vector[3] x;
  if (z)
    y ~ normal(0,1);
  else if (x)               // ERROR HERE
    y ~ normal(0,1);
}
