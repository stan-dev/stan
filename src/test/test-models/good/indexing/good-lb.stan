transformed data {
  int is[3];
  real a[4];
  real b[3];

  // OK
  a[3:] <- b[2:];
  a[:2] <- b[:2];
  a[3:4] <- b[2:3];
  a[is] <- b;
  a[:] <- a;
  a[ ] <- a[:];


}
parameters {
  real y;
}
model {
  y ~ normal(0, 1);
}
