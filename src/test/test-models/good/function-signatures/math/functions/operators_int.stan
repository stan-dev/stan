Info: Found int division at 'src/test/test-models/good/function-signatures/math/functions/operators_int.stan', line 10, column 25 to column 30:
  d_int / d_int
Values will be rounded towards zero. If rounding is not desired you can write
the division as
  d_int * 1.0 / d_int
If rounding is intended please use the integer division operator %/%.
Info: Found int division at 'src/test/test-models/good/function-signatures/math/functions/operators_int.stan', line 24, column 27 to column 32:
  d_int / d_int
Values will be rounded towards zero. If rounding is not desired you can write
the division as
  d_int * 1.0 / d_int
data {
  int d_int;
}
transformed data {
  int transformed_data_int;
  transformed_data_int = d_int + d_int;
  transformed_data_int = d_int - d_int;
  transformed_data_int = d_int * d_int;
  transformed_data_int = d_int / d_int;
  transformed_data_int = -d_int;
  transformed_data_int = +d_int;
}
parameters {
  real y_p;
}
transformed parameters {
  real transformed_param_real;
  transformed_param_real = d_int + d_int;
  transformed_param_real = d_int - d_int;
  transformed_param_real = d_int * d_int;
  transformed_param_real = d_int / d_int;
  transformed_param_real = -d_int;
  transformed_param_real = +d_int;
}
model {
  y_p ~ normal(0, 1);
}

If rounding is intended please use the integer division operator %/%.