Info: Found int division at 'src/test/test-models/good/validate_division_good.stan', line 19, column 7 to column 8:
  2 / 3
Values will be rounded towards zero. If rounding is not desired you can write
the division as
  2.0 / 3
transformed data {
  real u;
  matrix[3, 3] m;
  row_vector[3] v;
  u = 2.1 / 3;
  u = 2 / 3.1;
  u = 2.1 / 3.1;
  m = m / m;
  v = v / m;
}
parameters {
  real y;
}
transformed parameters {
  real xt;
  real ut;
  matrix[3, 3] mt;
  row_vector[3] vt;
  xt = 2 / 3;
  ut = 2.1 / 3;
  ut = 2 / 3.1;
  ut = 2.1 / 3.1;
  mt = mt / mt;
  vt = vt / mt;
}
model {
  y ~ normal(0, 1);
}

If rounding is intended please use the integer division operator %/%.