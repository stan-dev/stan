// compiles, should throw run-time error, mismatched sizes
parameters {
  real x;
}
model {
  matrix[2,3] z = [[1, 2, 3, 4], [4, 5, 6, 8]];
  x ~ normal(0, 5);
}
