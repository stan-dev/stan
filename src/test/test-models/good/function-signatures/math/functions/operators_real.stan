Info: Found int division at 'src/test/test-models/good/function-signatures/math/functions/operators_real.stan', line 23, column 26 to column 31:
  d_int / d_int
Values will be rounded towards zero. If rounding is not desired you can write
the division as
  d_int * 1.0 / d_int
If rounding is intended please use the integer division operator %/%.
Info: Found int division at 'src/test/test-models/good/function-signatures/math/functions/operators_real.stan', line 67, column 27 to column 32:
  d_int / d_int
Values will be rounded towards zero. If rounding is not desired you can write
the division as
  d_int * 1.0 / d_int
data {
  int d_int;
  real d_real;
}
transformed data {
  real transformed_data_real;
  transformed_data_real = d_real + d_real;
  transformed_data_real = d_real + d_int;
  transformed_data_real = d_int + d_real;
  transformed_data_real = d_int + d_int;
  transformed_data_real = d_real - d_real;
  transformed_data_real = d_int - d_real;
  transformed_data_real = d_real - d_int;
  transformed_data_real = d_int - d_int;
  transformed_data_real = d_real * d_real;
  transformed_data_real = d_int * d_real;
  transformed_data_real = d_real * d_int;
  transformed_data_real = d_int * d_int;
  transformed_data_real = d_real / d_real;
  transformed_data_real = d_int / d_real;
  transformed_data_real = d_real / d_int;
  transformed_data_real = d_int / d_int;
  transformed_data_real = -d_real;
  transformed_data_real = -d_int;
  transformed_data_real = +d_real;
  transformed_data_real = +d_int;
}
parameters {
  real p_real;
  real y_p;
}
transformed parameters {
  real transformed_param_real;
  transformed_param_real = d_real + d_real;
  transformed_param_real = d_real + d_int;
  transformed_param_real = d_int + d_real;
  transformed_param_real = d_int + d_int;
  transformed_param_real = p_real + d_real;
  transformed_param_real = p_real + d_int;
  transformed_param_real = d_real + p_real;
  transformed_param_real = d_int + p_real;
  transformed_param_real = p_real + p_real;
  transformed_param_real = d_real - d_real;
  transformed_param_real = d_real - d_int;
  transformed_param_real = d_int - d_real;
  transformed_param_real = d_int - d_int;
  transformed_param_real = p_real - d_real;
  transformed_param_real = p_real - d_int;
  transformed_param_real = d_real - p_real;
  transformed_param_real = d_int - p_real;
  transformed_param_real = p_real - p_real;
  transformed_param_real = d_real * d_real;
  transformed_param_real = d_real * d_int;
  transformed_param_real = d_int * d_real;
  transformed_param_real = d_int * d_int;
  transformed_param_real = p_real * d_real;
  transformed_param_real = p_real * d_int;
  transformed_param_real = d_real * p_real;
  transformed_param_real = d_int * p_real;
  transformed_param_real = p_real * p_real;
  transformed_param_real = d_real / d_real;
  transformed_param_real = d_real / d_int;
  transformed_param_real = d_int / d_real;
  transformed_param_real = d_int / d_int;
  transformed_param_real = p_real / d_real;
  transformed_param_real = p_real / d_int;
  transformed_param_real = d_real / p_real;
  transformed_param_real = d_int / p_real;
  transformed_param_real = p_real / p_real;
  transformed_param_real = -d_int;
  transformed_param_real = -d_real;
  transformed_param_real = -p_real;
  transformed_param_real = +d_int;
  transformed_param_real = +d_real;
  transformed_param_real = +p_real;
}
model {
  y_p ~ normal(0, 1);
}

If rounding is intended please use the integer division operator %/%.