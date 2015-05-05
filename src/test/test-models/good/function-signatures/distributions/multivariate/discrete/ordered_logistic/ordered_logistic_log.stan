data { 
  int d_int;
  real d_real;
  vector[d_int] d_vector;
}
transformed data {
  real transformed_data_real;

  transformed_data_real <- ordered_logistic_log(d_int, d_real, d_vector);
}
parameters {
  real p_real;
  vector[d_int] p_vector;

  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- ordered_logistic_log(d_int, d_real, d_vector);
  transformed_param_real <- ordered_logistic_log(d_int, p_real, d_vector);
  transformed_param_real <- ordered_logistic_log(d_int, d_real, p_vector);
  transformed_param_real <- ordered_logistic_log(d_int, p_real, p_vector);
}
model {  
  y_p ~ normal(0,1);
}
