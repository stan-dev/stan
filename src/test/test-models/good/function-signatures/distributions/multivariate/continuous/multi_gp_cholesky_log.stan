data {
  int d_int;
  matrix[d_int,d_int] d_matrix;
  cholesky_factor_cov[d_int] d_cov;
  vector[d_int] d_vector;
}
transformed data {
  real transformed_data_real;

  transformed_data_real <- multi_gp_cholesky_log(d_matrix, d_cov, d_vector);
  transformed_data_real <- multi_gp_cholesky_log(d_matrix, d_matrix, d_vector);
}
parameters {
  matrix[d_int,d_int] p_matrix;
  cholesky_factor_cov[d_int] p_cov;
  vector[d_int] p_vector;
  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- multi_gp_cholesky_log(d_matrix, d_matrix, d_vector);
  transformed_param_real <- multi_gp_cholesky_log(p_matrix, d_matrix, d_vector);
  transformed_param_real <- multi_gp_cholesky_log(d_matrix, p_matrix, d_vector);
  transformed_param_real <- multi_gp_cholesky_log(d_matrix, d_matrix, p_vector);
  transformed_param_real <- multi_gp_cholesky_log(p_matrix, p_matrix, d_vector);
  transformed_param_real <- multi_gp_cholesky_log(p_matrix, d_matrix, p_vector);
  transformed_param_real <- multi_gp_cholesky_log(p_matrix, p_matrix, p_vector);
  transformed_param_real <- multi_gp_cholesky_log(d_matrix, p_matrix, p_vector);
  transformed_param_real <- multi_gp_cholesky_log(d_matrix, d_cov, d_vector);
  transformed_param_real <- multi_gp_cholesky_log(p_matrix, d_cov, d_vector);
  transformed_param_real <- multi_gp_cholesky_log(d_matrix, p_cov, d_vector);
  transformed_param_real <- multi_gp_cholesky_log(d_matrix, d_cov, p_vector);
  transformed_param_real <- multi_gp_cholesky_log(p_matrix, p_cov, d_vector);
  transformed_param_real <- multi_gp_cholesky_log(p_matrix, d_cov, p_vector);
  transformed_param_real <- multi_gp_cholesky_log(p_matrix, p_cov, p_vector);
  transformed_param_real <- multi_gp_cholesky_log(d_matrix, p_cov, p_vector);
}
model {  
  d_matrix ~ multi_gp_cholesky(p_matrix, p_vector);
  d_matrix ~ multi_gp_cholesky(p_cov, p_vector); 
}
