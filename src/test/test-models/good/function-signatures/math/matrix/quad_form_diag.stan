data { 
  int d_int;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
  matrix[d_int,d_int] d_matrix;
}

transformed data {
  matrix[d_int,d_int] transformed_data_matrix;

  transformed_data_matrix <- quad_form_diag(d_matrix, d_vector);
  transformed_data_matrix <- quad_form_diag(d_matrix, d_row_vector);
}
parameters {
  real y_p;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
  matrix[d_int,d_int] p_matrix;
}
transformed parameters {
  matrix[d_int,d_int] transformed_param_matrix;

  transformed_param_matrix <- quad_form_diag(d_matrix, d_vector);
  transformed_param_matrix <- quad_form_diag(d_matrix, d_row_vector);

  transformed_param_matrix <- quad_form_diag(p_matrix, d_vector);
  transformed_param_matrix <- quad_form_diag(p_matrix, d_row_vector);

  transformed_param_matrix <- quad_form_diag(d_matrix, p_vector);
  transformed_param_matrix <- quad_form_diag(d_matrix, p_row_vector);

  transformed_param_matrix <- quad_form_diag(p_matrix, p_vector);
  transformed_param_matrix <- quad_form_diag(p_matrix, p_row_vector);
}
model {  
  y_p ~ normal(0,1);
}
