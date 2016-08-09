data { 
  int d_int_1;
  int d_int_2;
  int K_int;
  real d_arr_1[d_int_1];
  real d_arr_2[d_int_2];
  vector[K_int] d_vec_1[d_int_1];
  vector[K_int] d_vec_2[d_int_2];
  row_vector[K_int] d_rvec_1[d_int_1];
  row_vector[K_int] d_rvec_2[d_int_2];
  real d_sigma;
  real d_len;
}

transformed data {
  matrix[d_int,d_int] transformed_data_matrix;

  transformed_data_matrix <- cov_exp_quad(d_arr_1, d_sigma, d_real);
  transformed_data_matrix <- cov_exp_quad(d_arr_1, d_arr_2, d_sigma, d_real);
  transformed_data_matrix <- cov_exp_quad(d_vec_1, d_sigma, d_real);
  transformed_data_matrix <- cov_exp_quad(d_vec_1, d_vec_2, d_sigma, d_real);
  transformed_data_matrix <- cov_exp_quad(d_rvec_1, d_sigma, d_real);
  transformed_data_matrix <- cov_exp_quad(d_rvec_1, d_rvec_2, d_sigma, d_real);
}
parameters {
  real y_p;
  real p_arr_1[d_int_1];
  real p_arr_2[d_int_2];
  vector[K_int] p_vec_1[d_int_1];
  vector[K_int] p_vec_2[d_int_2];
  row_vector[K_int] p_rvec_1[d_int_1];
  row_vector[K_int] p_rvec_2[d_int_2];
  real p_sigma;
  real p_len;
  matrix[d_int,d_int] p_matrix;
}
transformed parameters {
  matrix[d_int,d_int] transformed_param_matrix;

  transformed_param_matrix <- cov_exp_quad(p_arr_1, p_sigma, p_real);
  transformed_param_matrix <- cov_exp_quad(p_arr_1, p_arr_2, p_sigma, p_real);
  transformed_param_matrix <- cov_exp_quad(p_vec_1, p_sigma, p_real);
  transformed_param_matrix <- cov_exp_quad(p_vec_1, p_vec_2, p_sigma, p_real);
  transformed_param_matrix <- cov_exp_quad(p_rvec_1, p_sigma, p_real);
  transformed_param_matrix <- cov_exp_quad(p_rvec_1, p_rvec_2, p_sigma, p_real);
}
model {  
  y_p ~ normal(0,1);
}
