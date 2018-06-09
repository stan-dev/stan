data {
  int d_int;
  int d_col_b;
  matrix[d_int,d_int] d_matrix;
  real d_t;
  matrix[d_int,d_col_b] d_matrix_b;
}
transformed data {
  matrix[d_int,d_int] transformed_data_matrix;
  transformed_data_matrix = matrix_exp(d_matrix);
  transformed_data_matrix = matrix_exp(d_matrix, d_matrix_b, d_t);
}
parameters {
  real y_p;
  matrix[d_int,d_int] p_matrix;
  real p_t;
  matrix[d_int,d_col_b] p_matrix_b;
}
transformed parameters {
  matrix[d_int,d_int] transformed_param_matrix;
  transformed_param_matrix = matrix_exp(p_matrix);
  transformed_param_matrix = matrix_exp(d_matrix, p_matrix_b, p_t);
  transformed_param_matrix = matrix_exp(d_matrix, d_matrix_b, p_t);
  transformed_param_matrix = matrix_exp(p_matrix, p_matrix_b, p_t);
  transformed_param_matrix = matrix_exp(p_matrix, d_matrix_b, p_t);
  transformed_param_matrix = matrix_exp(p_matrix, d_matrix_b, d_t);
}
model {
  y_p ~ normal(0,1);
}
