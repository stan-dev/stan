Info: Found int division at 'src/test/test-models/good/validate_division_int_warning.stan', line 7, column 6 to column 7:
  j / k
Values will be rounded towards zero. If rounding is not desired you can write
the division as
  j * 1.0 / k
transformed data {
  real u;
  int j;
  int k;
  j = 2;
  k = 3;
  u = j / k;
}
parameters {
  real y;
}
model {
  y ~ normal(0, 1);
}

If rounding is intended please use the integer division operator %/%.