parameters {
  real y;
}
model {
  vector[3] x;
  if (x)                   // ERROR HERE
    y ~ normal(0,1);
}
