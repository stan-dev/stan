data { 
  int d_int;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
  matrix[d_int,d_int] d_matrix;
}

transformed data {
  matrix[d_int,d_int] transformed_data_matrix;

  transformed_data_matrix <- diag_pre_multiply(d_vector, d_matrix);
  transformed_data_matrix <- diag_pre_multiply(d_row_vector, d_matrix);
}
parameters {
  real y_p;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
  matrix[d_int,d_int] p_matrix;
}
transformed parameters {
  matrix[d_int,d_int] transformed_param_matrix;

  transformed_param_matrix <- diag_pre_multiply(d_vector, d_matrix);
  transformed_param_matrix <- diag_pre_multiply(d_row_vector, d_matrix);

  transformed_param_matrix <- diag_pre_multiply(p_vector, d_matrix);
  transformed_param_matrix <- diag_pre_multiply(p_row_vector, d_matrix);

  transformed_param_matrix <- diag_pre_multiply(d_vector, p_matrix);
  transformed_param_matrix <- diag_pre_multiply(d_row_vector, p_matrix);

  transformed_param_matrix <- diag_pre_multiply(p_vector, p_matrix);
  transformed_param_matrix <- diag_pre_multiply(p_row_vector, p_matrix);
}
model {  
  y_p ~ normal(0,1);
}
