data { 
  int d_int;
  matrix[d_int,d_int] d_matrix;
  row_vector[d_int] d_row_vector;
  vector[d_int] d_vector;
}

transformed data {
  matrix[d_int,d_int] transformed_data_matrix;
  vector[d_int] transformed_data_vector;

  transformed_data_matrix <- append_row(d_matrix, d_matrix);
  transformed_data_matrix <- append_row(d_row_vector, d_matrix);
  transformed_data_matrix <- append_row(d_matrix, d_row_vector);
  transformed_data_matrix <- append_row(d_row_vector, d_row_vector);
  transformed_data_vector <- append_row(d_vector, d_vector);
}
parameters {
  real y_p;
  matrix[d_int,d_int] p_matrix;
  row_vector[d_int] p_row_vector;
  vector[d_int] p_vector;
}
transformed parameters {
  matrix[d_int,d_int] transformed_param_matrix;
  vector[d_int] transformed_param_vector;

  transformed_param_matrix <- append_row(p_matrix, d_matrix);
  transformed_param_matrix <- append_row(d_matrix, p_matrix);
  transformed_param_matrix <- append_row(p_matrix, p_matrix);
  transformed_param_matrix <- append_row(d_matrix, d_matrix);

  transformed_param_matrix <- append_row(p_row_vector, d_matrix);
  transformed_param_matrix <- append_row(d_row_vector, p_matrix);
  transformed_param_matrix <- append_row(p_row_vector, p_matrix);
  transformed_param_matrix <- append_row(d_row_vector, d_matrix);
  
  transformed_param_matrix <- append_row(p_matrix, d_row_vector);
  transformed_param_matrix <- append_row(d_matrix, p_row_vector);
  transformed_param_matrix <- append_row(p_matrix, p_row_vector);
  transformed_param_matrix <- append_row(d_matrix, d_row_vector);
  
  transformed_param_matrix <- append_row(p_row_vector, d_row_vector);
  transformed_param_matrix <- append_row(d_row_vector, p_row_vector);
  transformed_param_matrix <- append_row(p_row_vector, p_row_vector);
  transformed_param_matrix <- append_row(d_row_vector, d_row_vector);
  
  transformed_param_vector <- append_row(p_vector, d_vector);
  transformed_param_vector <- append_row(d_vector, p_vector);
  transformed_param_vector <- append_row(p_vector, p_vector);
  transformed_param_vector <- append_row(d_vector, d_vector);
}
model {  
  y_p ~ normal(0,1);
}
