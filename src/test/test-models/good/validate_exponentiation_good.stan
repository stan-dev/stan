data {
  int i;
  int j;
  real x;
  real y;
}
transformed data {
  real z;
  z <- x ^ y;  // double, double
  z <- x ^ j;  // double, int
  z <- j ^ x;  // int, double
  z <- i ^ j;  // int, int
}
parameters {
  real a;
  real b;
}
transformed parameters {
  real z2;
  z2 <- a * b;
  z2 <- x ^ y;  // double, double
  z2 <- x ^ j;  // double, int
  z2 <- x ^ a;  // double, var
  z2 <- i ^ x;  // int, double
  z2 <- i ^ j;  // int, int
  z2 <- i ^ b;  // int, var
  z2 <- a ^ x;  // var, double
  z2 <- a ^ j;  // var, int
  z2 <- a ^ b;  // var, var
}
model {
  a ~ normal(0,1);
}
