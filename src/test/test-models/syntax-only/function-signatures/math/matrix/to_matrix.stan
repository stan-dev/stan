data { 
  int d_int;
  int d_int_array_2[d_int, d_int];
  real d_real_array_2[d_int, d_int];
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}

transformed data {
  matrix[d_int,d_int] transformed_data_matrix;

  transformed_data_matrix <- to_matrix(d_matrix);
  transformed_data_matrix <- to_matrix(d_vector);
  transformed_data_matrix <- to_matrix(d_row_vector);
  transformed_data_matrix <- to_matrix(d_int_array_2);
  transformed_data_matrix <- to_matrix(d_real_array_2);
}
parameters {
  real p_real;
  real y_p;
  real p_real_array_2[d_int, d_int];
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
}
transformed parameters {
  matrix[d_int,d_int] transformed_param_matrix;

  transformed_param_matrix <- to_matrix(d_matrix);
  transformed_param_matrix <- to_matrix(d_vector);
  transformed_param_matrix <- to_matrix(d_row_vector);
  transformed_param_matrix <- to_matrix(d_int_array_2);
  transformed_param_matrix <- to_matrix(d_real_array_2);
  transformed_param_matrix <- to_matrix(p_matrix);
  transformed_param_matrix <- to_matrix(p_vector);
  transformed_param_matrix <- to_matrix(p_row_vector);
  transformed_param_matrix <- to_matrix(p_real_array_2);
}
model {  
  y_p ~ normal(0,1);
}
