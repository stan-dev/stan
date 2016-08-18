data { 
  int d_int;
  real d_real;
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}

transformed data {
  matrix[d_int,d_int] transformed_data_matrix;
  row_vector[d_int] transformed_data_row_vector;

  transformed_data_matrix <- append_col(d_matrix, d_matrix);
  transformed_data_matrix <- append_col(d_vector, d_matrix);
  transformed_data_matrix <- append_col(d_matrix, d_vector);
  transformed_data_matrix <- append_col(d_vector, d_vector);
  transformed_data_row_vector <- append_col(d_row_vector, d_row_vector);
  transformed_data_row_vector <- append_col(d_real, d_row_vector);
  transformed_data_row_vector <- append_col(d_row_vector, d_real);
}
parameters {
  real p_real;
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
}
transformed parameters {
  matrix[d_int,d_int] transformed_param_matrix;
  row_vector[d_int] transformed_param_row_vector;

  transformed_param_matrix <- append_col(p_matrix, d_matrix);
  transformed_param_matrix <- append_col(d_matrix, p_matrix);
  transformed_param_matrix <- append_col(p_matrix, p_matrix);
  transformed_param_matrix <- append_col(d_matrix, d_matrix);

  transformed_param_matrix <- append_col(p_vector, d_matrix);
  transformed_param_matrix <- append_col(d_vector, p_matrix);
  transformed_param_matrix <- append_col(p_vector, p_matrix);
  transformed_param_matrix <- append_col(d_vector, d_matrix);
  
  transformed_param_matrix <- append_col(p_matrix, d_vector);
  transformed_param_matrix <- append_col(d_matrix, p_vector);
  transformed_param_matrix <- append_col(p_matrix, p_vector);
  transformed_param_matrix <- append_col(d_matrix, d_vector);
  
  transformed_param_matrix <- append_col(p_vector, d_vector);
  transformed_param_matrix <- append_col(d_vector, p_vector);
  transformed_param_matrix <- append_col(p_vector, p_vector);
  transformed_param_matrix <- append_col(d_vector, d_vector);
  
  transformed_param_row_vector <- append_col(p_row_vector, d_row_vector);
  transformed_param_row_vector <- append_col(d_row_vector, p_row_vector);
  transformed_param_row_vector <- append_col(p_row_vector, p_row_vector);
  transformed_param_row_vector <- append_col(d_row_vector, d_row_vector);
  
  transformed_param_row_vector <- append_col(p_real, d_row_vector);
  transformed_param_row_vector <- append_col(d_real, p_row_vector);
  transformed_param_row_vector <- append_col(p_real, p_row_vector);
  transformed_param_row_vector <- append_col(d_real, d_row_vector);
  
  transformed_param_row_vector <- append_col(p_row_vector, d_real);
  transformed_param_row_vector <- append_col(d_row_vector, p_real);
  transformed_param_row_vector <- append_col(p_row_vector, p_real);
  transformed_param_row_vector <- append_col(d_row_vector, d_real);
}
model {  
  p_real ~ normal(0,1);
}
