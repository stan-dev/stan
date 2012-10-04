data { 
  int d_int;
  int d_int_array[d_int];
  real d_real;
  real d_real_array[d_int];
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data {
  int transformed_data_int;
  real transformed_data_real;
  real transformed_data_real_array[d_int];
  matrix[d_int,d_int] transformed_data_matrix;
  vector[d_int] transformed_data_vector;
  row_vector[d_int] transformed_data_row_vector;

  // Vector Probabilities
  transformed_data_real <- multi_normal_log(d_vector, d_vector, d_matrix);
  transformed_data_real <- multi_normal_cholesky_log(d_vector, d_vector, d_matrix);
  transformed_data_real <- multi_student_t_log(d_vector, d_real, d_vector, d_matrix);
  // Covariance Matrix Distributions
  transformed_data_real <- wishart_log(d_matrix, d_real, d_matrix);
  transformed_data_real <- inv_wishart_log(d_matrix, d_real, d_matrix);
  transformed_data_real <- lkj_cov_log(d_matrix, d_vector, d_vector, d_real);
  transformed_data_real <- lkj_corr_log(d_matrix, d_real);
  transformed_data_real <- lkj_corr_cholesky_log(d_matrix, d_real); 
  // mdivide_..._low
  transformed_data_matrix <- mdivide_left_tri_low(d_matrix,d_matrix);
  transformed_data_vector <- mdivide_left_tri_low(d_matrix,d_vector);
  transformed_data_matrix <- mdivide_right_tri_low(d_matrix,d_matrix);
  transformed_data_row_vector <- mdivide_right_tri_low(d_row_vector,d_matrix);
}
parameters {
  real p_real;
  real p_real_array[d_int];
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
}
transformed parameters {
  real transformed_param_real;
  real transformed_param_real_array[d_int];
  matrix[d_int,d_int] transformed_param_matrix;
  vector[d_int] transformed_param_vector;
  row_vector[d_int] transformed_param_row_vector;

  // Vector Probabilities
  transformed_param_real <- multi_normal_log(d_vector, d_vector, d_matrix);
  transformed_param_real <- multi_normal_log(d_vector, d_vector, p_matrix);
  transformed_param_real <- multi_normal_log(d_vector, p_vector, d_matrix);
  transformed_param_real <- multi_normal_log(d_vector, p_vector, p_matrix);
  transformed_param_real <- multi_normal_log(p_vector, d_vector, d_matrix);
  transformed_param_real <- multi_normal_log(p_vector, d_vector, p_matrix);
  transformed_param_real <- multi_normal_log(p_vector, p_vector, d_matrix);
  transformed_param_real <- multi_normal_log(p_vector, p_vector, p_matrix);
  transformed_param_real <- multi_normal_cholesky_log(d_vector, d_vector, d_matrix);
  transformed_param_real <- multi_normal_cholesky_log(d_vector, d_vector, p_matrix);
  transformed_param_real <- multi_normal_cholesky_log(d_vector, p_vector, d_matrix);
  transformed_param_real <- multi_normal_cholesky_log(d_vector, p_vector, p_matrix);
  transformed_param_real <- multi_normal_cholesky_log(p_vector, d_vector, d_matrix);
  transformed_param_real <- multi_normal_cholesky_log(p_vector, d_vector, p_matrix);
  transformed_param_real <- multi_normal_cholesky_log(p_vector, p_vector, d_matrix);
  transformed_param_real <- multi_normal_cholesky_log(p_vector, p_vector, p_matrix);
  // Covariance Matrix Distributions
  transformed_param_real <- wishart_log(d_matrix, d_real, d_matrix);
  transformed_param_real <- wishart_log(d_matrix, d_real, p_matrix);
  transformed_param_real <- wishart_log(d_matrix, p_real, d_matrix);
  transformed_param_real <- wishart_log(d_matrix, p_real, p_matrix);
  transformed_param_real <- wishart_log(p_matrix, d_real, d_matrix);
  transformed_param_real <- wishart_log(p_matrix, d_real, p_matrix);
  transformed_param_real <- wishart_log(p_matrix, p_real, d_matrix);
  transformed_param_real <- wishart_log(p_matrix, p_real, p_matrix);
  transformed_param_real <- inv_wishart_log(d_matrix, d_real, d_matrix);
  transformed_param_real <- inv_wishart_log(d_matrix, d_real, p_matrix);
  transformed_param_real <- inv_wishart_log(d_matrix, p_real, d_matrix);
  transformed_param_real <- inv_wishart_log(d_matrix, p_real, p_matrix);
  transformed_param_real <- inv_wishart_log(p_matrix, d_real, d_matrix);
  transformed_param_real <- inv_wishart_log(p_matrix, d_real, p_matrix);
  transformed_param_real <- inv_wishart_log(p_matrix, p_real, d_matrix);
  transformed_param_real <- inv_wishart_log(p_matrix, p_real, p_matrix);
  transformed_param_real <- lkj_cov_log(d_matrix, d_vector, d_vector, d_real);
  transformed_param_real <- lkj_cov_log(d_matrix, d_vector, d_vector, p_real);
  transformed_param_real <- lkj_cov_log(d_matrix, d_vector, p_vector, d_real);
  transformed_param_real <- lkj_cov_log(d_matrix, d_vector, p_vector, p_real);
  transformed_param_real <- lkj_cov_log(d_matrix, p_vector, d_vector, d_real);
  transformed_param_real <- lkj_cov_log(d_matrix, p_vector, d_vector, p_real);
  transformed_param_real <- lkj_cov_log(d_matrix, p_vector, p_vector, d_real);
  transformed_param_real <- lkj_cov_log(d_matrix, p_vector, p_vector, p_real);
  transformed_param_real <- lkj_cov_log(p_matrix, d_vector, d_vector, d_real);
  transformed_param_real <- lkj_cov_log(p_matrix, d_vector, d_vector, p_real);
  transformed_param_real <- lkj_cov_log(p_matrix, d_vector, p_vector, d_real);
  transformed_param_real <- lkj_cov_log(p_matrix, d_vector, p_vector, p_real);
  transformed_param_real <- lkj_cov_log(p_matrix, p_vector, d_vector, d_real);
  transformed_param_real <- lkj_cov_log(p_matrix, p_vector, d_vector, p_real);
  transformed_param_real <- lkj_cov_log(p_matrix, p_vector, p_vector, d_real);
  transformed_param_real <- lkj_cov_log(p_matrix, p_vector, p_vector, p_real);
  transformed_param_real <- lkj_corr_log(d_matrix, d_real);
  transformed_param_real <- lkj_corr_log(d_matrix, p_real);
  transformed_param_real <- lkj_corr_log(p_matrix, d_real);
  transformed_param_real <- lkj_corr_log(p_matrix, p_real);

  // mdivide_..._low
  transformed_param_matrix <- mdivide_left_tri_low(d_matrix,d_matrix);
  transformed_param_matrix <- mdivide_left_tri_low(transformed_param_matrix,d_matrix);
  transformed_param_matrix <- mdivide_left_tri_low(transformed_param_matrix,
                                                   transformed_param_matrix);
  transformed_param_matrix <- mdivide_left_tri_low(d_matrix,
                                                   transformed_param_matrix);

  transformed_param_vector <- mdivide_left_tri_low(d_matrix,d_vector);
  transformed_param_vector <- mdivide_left_tri_low(transformed_param_matrix,d_vector);
  transformed_param_vector <- mdivide_left_tri_low(transformed_param_matrix,
                                                   transformed_param_vector);
  transformed_param_vector <- mdivide_left_tri_low(d_matrix,
                                                   transformed_param_vector);

  transformed_param_matrix <- mdivide_right_tri_low(d_matrix,d_matrix);
  transformed_param_matrix <- mdivide_right_tri_low(transformed_param_matrix,d_matrix);
  transformed_param_matrix <- mdivide_right_tri_low(transformed_param_matrix,
                                                    transformed_param_matrix);
  transformed_param_matrix <- mdivide_right_tri_low(d_matrix,
                                                    transformed_param_matrix);

  transformed_param_row_vector <- mdivide_right_tri_low(d_row_vector,d_matrix);
  transformed_param_row_vector <- mdivide_right_tri_low(transformed_param_row_vector,d_matrix);
  transformed_param_row_vector <- mdivide_right_tri_low(transformed_param_row_vector,
                                                        transformed_param_matrix);
  transformed_param_row_vector <- mdivide_right_tri_low(d_row_vector,
                                                        transformed_param_matrix);

}
model {  
}
