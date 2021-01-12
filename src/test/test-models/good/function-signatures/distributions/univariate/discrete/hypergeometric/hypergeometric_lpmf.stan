data {
  int d_int;
}
transformed data {
  real transformed_data_real;
  transformed_data_real = hypergeometric_log(d_int, d_int, d_int, d_int);
}
parameters {
  real y_p;
}
transformed parameters {
  real transformed_param_real;
  transformed_param_real = hypergeometric_log(d_int, d_int, d_int, d_int);
}
model {
  y_p ~ normal(0, 1);
}

