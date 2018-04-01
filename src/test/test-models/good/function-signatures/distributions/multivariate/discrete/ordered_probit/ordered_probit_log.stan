data { 
  int d_int;
  real d_real;
  int d_int_array[d_int];
  vector[d_int] d_vector;
  vector[d_int] d_vector_array[d_int];
}
transformed data {
  real transformed_data_real;

  transformed_data_real <- ordered_probit_log(d_int, d_real, d_vector);
  transformed_data_real <- ordered_probit_log(d_int_array, d_vector, d_vector);
  transformed_data_real <- ordered_probit_log(d_int_array, d_vector, d_vector_array);
}
parameters {
  real p_real;
  vector[d_int] p_vector;
  vector[d_int] p_vector_array[d_int];

  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- ordered_probit_log(d_int, d_real, d_vector);
  transformed_param_real <- ordered_probit_log(d_int_array, d_vector, d_vector);
  transformed_param_real <- ordered_probit_log(d_int_array, d_vector, d_vector_array);

  transformed_param_real <- ordered_probit_log(d_int, p_real, d_vector);
  transformed_param_real <- ordered_probit_log(d_int_array, p_vector, d_vector);
  transformed_param_real <- ordered_probit_log(d_int_array, p_vector, d_vector_array);

  transformed_param_real <- ordered_probit_log(d_int, d_real, p_vector);
  transformed_param_real <- ordered_probit_log(d_int_array, d_vector, p_vector);
  transformed_param_real <- ordered_probit_log(d_int_array, d_vector, p_vector_array);

  transformed_param_real <- ordered_probit_log(d_int, p_real, p_vector);
  transformed_param_real <- ordered_probit_log(d_int_array, p_vector, p_vector);
  transformed_param_real <- ordered_probit_log(d_int_array, p_vector, p_vector_array);
}
model {  
  y_p ~ normal(0,1);
}
