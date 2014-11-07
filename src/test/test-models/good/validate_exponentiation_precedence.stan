data {
}
transformed data {
  vector[3] a;
  vector[3] b;
  real c;
  int i;
  real x;
  a[1] <- 2.1;
  a[2] <- 2.2;
  a[3] <- 2.3;
  b[1] <- 2;
  b[2] <- 3;
  b[3] <- 4;
  i <- 5;
  x <- 6.66;
  c <- a[1] ^ b[1];
  c <- a[1] ^ x;
  c <- a[1] ^ i;
  c <- i ^ a[1];
  c <- x ^ a[1];
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
