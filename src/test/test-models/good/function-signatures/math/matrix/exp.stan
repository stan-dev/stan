data { 
  int d_int;
  int d_int_array[d_int];
  real d_real_array[d_int];
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}

transformed data {
  matrix[d_int,d_int] transformed_data_matrix;
  vector[d_int] transformed_data_vector;
  row_vector[d_int] transformed_data_row_vector;

  transformed_data_matrix <- exp(d_matrix);
  transformed_data_vector <- exp(d_vector);
  transformed_data_row_vector <- exp(d_row_vector);
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
  matrix[d_int,d_int] transformed_param_matrix;
  vector[d_int] transformed_param_vector;
  row_vector[d_int] transformed_param_row_vector;

  transformed_param_matrix <- exp(d_matrix);
  transformed_param_vector <- exp(d_vector);
  transformed_param_row_vector <- exp(d_row_vector);
  transformed_param_matrix <- exp(p_matrix);
  transformed_param_vector <- exp(p_vector);
  transformed_param_row_vector <- exp(p_row_vector);
}
model {  
  y_p ~ normal(0,1);
}
