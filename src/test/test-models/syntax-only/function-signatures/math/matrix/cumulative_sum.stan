data { 
  int d_int;
  int d_int_array[d_int];
  real d_real_array[d_int];
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}

transformed data {
  real transformed_data_real_array[d_int];
  vector[d_int] transformed_data_vector;
  row_vector[d_int] transformed_data_row_vector;

  transformed_data_real_array <- cumulative_sum(d_real_array);
  transformed_data_vector <- cumulative_sum(d_vector);
  transformed_data_row_vector <- cumulative_sum(d_row_vector);
}
parameters {
  real p_real;
  real y_p;
  real p_real_array[d_int];
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
}
transformed parameters {
  real transformed_param_real_array[d_int];
  vector[d_int] transformed_param_vector;
  row_vector[d_int] transformed_param_row_vector;

  transformed_param_real_array <- cumulative_sum(d_real_array);
  transformed_param_vector <- cumulative_sum(d_vector);
  transformed_param_row_vector <- cumulative_sum(d_row_vector);
  transformed_param_real_array <- cumulative_sum(p_real_array);
  transformed_param_vector <- cumulative_sum(p_vector);
  transformed_param_row_vector <- cumulative_sum(p_row_vector);
}
model {  
  y_p ~ normal(0,1);
}
