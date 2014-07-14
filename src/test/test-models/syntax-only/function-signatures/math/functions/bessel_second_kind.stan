data { 
  int  d_int;
  int  r_int;
  real d_real;
}
transformed data {
  int transformed_data_int;
  real transformed_data_real;

  transformed_data_real <- bessel_second_kind(d_int, r_int);
  transformed_data_real <- bessel_second_kind(d_int, d_real);
}
parameters {
  real p_real;
  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- bessel_second_kind(d_int, r_int);
  transformed_param_real <- bessel_second_kind(d_int, d_real);
  transformed_param_real <- bessel_second_kind(d_int, p_real);
}
model {  
  y_p ~ normal(0,1);
}