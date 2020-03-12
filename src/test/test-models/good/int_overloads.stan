data { 
  int d_int;
  real d_real;
}

transformed data {
  int transformed_data_int;
  real transformed_data_real;

  transformed_data_real = abs(d_int);
  transformed_data_real = abs(d_real);
  transformed_data_real = fabs(d_int);
  transformed_data_real = fabs(d_real);

  transformed_data_real = exp(d_int);
  transformed_data_real = exp(d_real);
  transformed_data_real = exp2(d_int);
  transformed_data_real = exp2(d_real);
  transformed_data_real = expm1(d_int);
  transformed_data_real = expm1(d_real);
  transformed_data_real = log(d_int);
  transformed_data_real = log(d_real);
  transformed_data_real = log1p(d_int);
  transformed_data_real = log1p(d_real);
  transformed_data_real = log2(d_int);
  transformed_data_real = log2(d_real);
  transformed_data_real = log10(d_int);
  transformed_data_real = log10(d_real);

  transformed_data_real = pow(d_int);
  transformed_data_real = pow(d_real);
  transformed_data_real = sqrt(d_int);
  transformed_data_real = sqrt(d_real);
  transformed_data_real = cbrt(d_int);
  transformed_data_real = cbrt(d_real);

  transformed_data_real = sin(d_int);
  transformed_data_real = sin(d_real);
  transformed_data_real = cos(d_int);
  transformed_data_real = cos(d_real);
  transformed_data_real = tan(d_int);
  transformed_data_real = tan(d_real);
  transformed_data_real = asin(d_int);
  transformed_data_real = asin(d_real);
  transformed_data_real = acos(d_int);
  transformed_data_real = acos(d_real);
  transformed_data_real = atan(d_int);
  transformed_data_real = atan(d_real);

  transformed_data_real = sinh(d_int);
  transformed_data_real = sinh(d_real);
  transformed_data_real = cosh(d_int);
  transformed_data_real = cosh(d_real);
  transformed_data_real = tanh(d_int);
  transformed_data_real = tanh(d_real);
  transformed_data_real = asinh(d_int);
  transformed_data_real = asinh(d_real);
  transformed_data_real = acosh(d_int);
  transformed_data_real = acosh(d_real);
  transformed_data_real = atanh(d_int);
  transformed_data_real = atanh(d_real);

  transformed_data_real = erf(d_int);
  transformed_data_real = erf(d_real);
  transformed_data_real = erfc(d_int);
  transformed_data_real = erfc(d_real);
  transformed_data_real = tgamma(d_int);
  transformed_data_real = tgamma(d_real);

  transformed_data_real = ceil(d_int);
  transformed_data_real = ceil(d_real);
  transformed_data_real = floor(d_int);
  transformed_data_real = floor(d_real);
  transformed_data_real = trunc(d_int);
  transformed_data_real = trunc(d_real);
  transformed_data_real = round(d_int);
  transformed_data_real = round(d_real);

  transformed_data_real = inv(d_int);
  transformed_data_real = inv(d_real);
  transformed_data_real = inv_sqrt(d_int);
  transformed_data_real = inv_sqrt(d_real);
  transformed_data_real = inv_square(d_int);
  transformed_data_real = inv_square(d_real);
  transformed_data_real = sqrt(d_int);
  transformed_data_real = sqrt(d_real);
  transformed_data_real = tan(d_int);
  transformed_data_real = tan(d_real);
  transformed_data_real = tanh(d_int);
  transformed_data_real = tanh(d_real);
  transformed_data_real = trigamma(d_int);
  transformed_data_real = trigamma(d_real);
}

parameters {
  real p_real;
  real y_p;
}
model {
  y ~ normal(0,1);
}

transformed parameters {
  real transformed_param_real;

  transformed_param_real = abs(d_int);
  transformed_param_real = abs(d_real);
  transformed_param_real = fabs(d_int);
  transformed_param_real = fabs(d_real);

  transformed_param_real = exp(d_int);
  transformed_param_real = exp(d_real);
  transformed_param_real = exp2(d_int);
  transformed_param_real = exp2(d_real);
  transformed_param_real = expm1(d_int);
  transformed_param_real = expm1(d_real);
  transformed_param_real = log(d_int);
  transformed_param_real = log(d_real);
  transformed_param_real = log1p(d_int);
  transformed_param_real = log1p(d_real);
  transformed_param_real = log2(d_int);
  transformed_param_real = log2(d_real);
  transformed_param_real = log10(d_int);
  transformed_param_real = log10(d_real);

  transformed_param_real = pow(d_int);
  transformed_param_real = pow(d_real);
  transformed_param_real = sqrt(d_int);
  transformed_param_real = sqrt(d_real);
  transformed_param_real = cbrt(d_int);
  transformed_param_real = cbrt(d_real);

  transformed_param_real = sin(d_int);
  transformed_param_real = sin(d_real);
  transformed_param_real = cos(d_int);
  transformed_param_real = cos(d_real);
  transformed_param_real = tan(d_int);
  transformed_param_real = tan(d_real);
  transformed_param_real = asin(d_int);
  transformed_param_real = asin(d_real);
  transformed_param_real = acos(d_int);
  transformed_param_real = acos(d_real);
  transformed_param_real = atan(d_int);
  transformed_param_real = atan(d_real);

  transformed_param_real = sinh(d_int);
  transformed_param_real = sinh(d_real);
  transformed_param_real = cosh(d_int);
  transformed_param_real = cosh(d_real);
  transformed_param_real = tanh(d_int);
  transformed_param_real = tanh(d_real);
  transformed_param_real = asinh(d_int);
  transformed_param_real = asinh(d_real);
  transformed_param_real = acosh(d_int);
  transformed_param_real = acosh(d_real);
  transformed_param_real = atanh(d_int);
  transformed_param_real = atanh(d_real);

  transformed_param_real = erf(d_int);
  transformed_param_real = erf(d_real);
  transformed_param_real = erfc(d_int);
  transformed_param_real = erfc(d_real);
  transformed_param_real = tgamma(d_int);
  transformed_param_real = tgamma(d_real);

  transformed_param_real = ceil(d_int);
  transformed_param_real = ceil(d_real);
  transformed_param_real = floor(d_int);
  transformed_param_real = floor(d_real);
  transformed_param_real = trunc(d_int);
  transformed_param_real = trunc(d_real);
  transformed_param_real = round(d_int);
  transformed_param_real = round(d_real);

  transformed_param_real = inv(d_int);
  transformed_param_real = inv(d_real);
  transformed_param_real = inv_sqrt(d_int);
  transformed_param_real = inv_sqrt(d_real);
  transformed_param_real = inv_square(d_int);
  transformed_param_real = inv_square(d_real);
  transformed_param_real = sqrt(d_int);
  transformed_param_real = sqrt(d_real);
  transformed_param_real = tan(d_int);
  transformed_param_real = tan(d_real);
  transformed_param_real = tanh(d_int);
  transformed_param_real = tanh(d_real);
  transformed_param_real = trigamma(d_int);
  transformed_param_real = trigamma(d_real);

}

model {  
  y_p ~ normal(0,1);
}
