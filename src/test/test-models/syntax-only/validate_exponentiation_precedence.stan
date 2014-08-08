transformed data {
  vector[3] a;
  vector[3] b;
  real c;
  c <- a[1] ^ b[1];
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
