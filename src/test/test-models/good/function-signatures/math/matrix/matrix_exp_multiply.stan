data {
  int d_int;
  int d_col;
  matrix[d_int,d_int] d_matrix_a;
  matrix[d_int,d_col] d_matrix_b;
}
transformed data {
  matrix[d_int,d_col] transformed_data_matrix;
  transformed_data_matrix = matrix_exp_multiply(d_matrix_a, d_matrix_b);
}
parameters {
  real y_p;
  matrix[d_int,d_int] p_matrix_a;
  matrix[d_int,d_col] p_matrix_b;
}
transformed parameters {
  matrix[d_int,d_col] transformed_param_matrix;
        
  transformed_param_matrix = matrix_exp_multiply(p_matrix_a, p_matrix_b);
  transformed_param_matrix = matrix_exp_multiply(p_matrix_a, d_matrix_b);
  transformed_param_matrix = matrix_exp_multiply(d_matrix_a, p_matrix_b);
}
model {
  y_p ~ normal(0,1);
}
