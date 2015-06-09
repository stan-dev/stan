parameters {
  matrix[max(1,3), min(2,5)] a;
}
transformed parameters {
  real z;
  z <- max(9,10);
  z <- min(9,10);
}
model {
  int b;

  // int
  b <- max(9, 10);

  to_vector(a) ~ normal(0,1);
}
