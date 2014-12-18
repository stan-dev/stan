data { 
  int d_int;
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data {
  row_vector[d_int] transformed_data_row_vector;

  transformed_data_row_vector <- columns_dot_product(d_vector, d_vector);
  transformed_data_row_vector <- columns_dot_product(d_row_vector, d_row_vector);
  transformed_data_row_vector <- columns_dot_product(d_matrix, d_matrix);
}
parameters {
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
  real y_p;
}
transformed parameters {
  row_vector[d_int] transformed_param_row_vector;

  transformed_param_row_vector <- columns_dot_product(d_vector, d_vector);
  transformed_param_row_vector <- columns_dot_product(d_row_vector, d_row_vector);
  transformed_param_row_vector <- columns_dot_product(d_matrix, d_matrix);

  transformed_param_row_vector <- columns_dot_product(p_vector, d_vector);
  transformed_param_row_vector <- columns_dot_product(p_row_vector, d_row_vector);
  transformed_param_row_vector <- columns_dot_product(p_matrix, d_matrix);

  transformed_param_row_vector <- columns_dot_product(d_vector, p_vector);
  transformed_param_row_vector <- columns_dot_product(d_row_vector, p_row_vector);
  transformed_param_row_vector <- columns_dot_product(d_matrix, p_matrix);

  transformed_param_row_vector <- columns_dot_product(p_vector, p_vector);
  transformed_param_row_vector <- columns_dot_product(p_row_vector, p_row_vector);
  transformed_param_row_vector <- columns_dot_product(p_matrix, p_matrix);
}
model {  
  y_p ~ normal(0,1);
}
