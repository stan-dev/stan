Info: Found int division at 'src/test/test-models/good/int_div_user.stan', line 7, column 6 to column 10:
  a[1] / b[2]
Values will be rounded towards zero. If rounding is not desired you can write
the division as
  a[1] * 1.0 / b[2]
data {
  array[4] int a;
  array[3] int b;
}
transformed data {
  int c;
  c = a[1] / b[2];
}
model {

}

If rounding is intended please use the integer division operator %/%.