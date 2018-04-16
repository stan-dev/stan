data { 
  int d_int;
  int d_int_array[d_int];
  real d_real;
  real d_real_array[d_int];
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data {
  real transformed_data_real;

  transformed_data_real <- std_normal_log(d_int);
  transformed_data_real <- std_normal_log(d_real);
  transformed_data_real <- std_normal_log(d_real_array);
  transformed_data_real <- std_normal_log(d_vector);
  transformed_data_real <- std_normal_log(d_row_vector);
}
parameters {
  real p_real;
  real p_real_array[d_int];
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;

  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- std_normal_log(d_int);
  transformed_param_real <- std_normal_log(d_real);
  transformed_param_real <- std_normal_log(d_real_array);
  transformed_param_real <- std_normal_log(d_vector);
  transformed_param_real <- std_normal_log(d_row_vector);
  transformed_param_real <- std_normal_log(p_real);
  transformed_param_real <- std_normal_log(p_real_array);
  transformed_param_real <- std_normal_log(p_vector);
  transformed_param_real <- std_normal_log(p_row_vector);
}
model {  
  y_p ~ normal(0,1); // in case anyone tries to run it
}
