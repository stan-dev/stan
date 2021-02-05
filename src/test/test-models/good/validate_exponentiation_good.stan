data {
  int i;
  int j;
  real x;
  real y;
}
transformed data {
  real z;
  z = x ^ y;
  z = x ^ j;
  z = j ^ x;
  z = i ^ j;
}
parameters {
  real a;
  real b;
}
transformed parameters {
  real z2;
  z2 = a * b;
  z2 = x ^ y;
  z2 = x ^ j;
  z2 = x ^ a;
  z2 = i ^ x;
  z2 = i ^ j;
  z2 = i ^ b;
  z2 = a ^ x;
  z2 = a ^ j;
  z2 = a ^ b;
}
model {
  a ~ normal(0, 1);
}

