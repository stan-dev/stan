// compiles, should throw run-time error, mismatched sizes
functions {
  real foo(real a1) {
    matrix[2,3] lf_m23 = [[1, 2, 3, 4], [4, 5, 6, 7]];
    real b;
    b += a1;
    return b;
  }
}
transformed data {
  real a1 = 1.0;
  real bar = foo(a1);
}
