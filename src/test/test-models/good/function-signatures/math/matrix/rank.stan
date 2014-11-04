data { 
  int d_int;
  int d_int_array[d_int];
  real d_real_array[d_int];
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}

transformed data {
  int transformed_data_int;
  real transformed_data_real;

  transformed_data_int <- rank(d_int_array, d_int);
  transformed_data_int <- rank(d_real_array, d_int);
  transformed_data_int <- rank(d_vector, d_int);
  transformed_data_int <- rank(d_row_vector, d_int);

  transformed_data_real <- rank(d_int_array, d_int);
  transformed_data_real <- rank(d_real_array, d_int);
  transformed_data_real <- rank(d_vector, d_int);
  transformed_data_real <- rank(d_row_vector, d_int);
}
parameters {
  real y_p;
  real p_real_array[d_int];
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- rank(d_int_array, d_int);
  transformed_param_real <- rank(d_real_array, d_int);
  transformed_param_real <- rank(d_vector, d_int);
  transformed_param_real <- rank(d_row_vector, d_int);

  transformed_param_real <- rank(p_real_array, d_int);
  transformed_param_real <- rank(p_vector, d_int);
  transformed_param_real <- rank(p_row_vector, d_int);
}
model {  
  y_p ~ normal(0,1);
}
