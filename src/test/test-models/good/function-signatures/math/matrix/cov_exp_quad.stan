data {
  int d_int_1;
  int d_int_2;
  int K;
  real d_sigma;
  real d_len;
  array[d_int_1] real d_arr_1;
  array[d_int_2] real d_arr_2;
  array[d_int_1] vector[K] d_vec_1;
  array[d_int_2] vector[K] d_vec_2;
  array[d_int_1] row_vector[K] d_rvec_1;
  array[d_int_2] row_vector[K] d_rvec_2;
}
transformed data {
  matrix[d_int_1, d_int_1] transformed_data_matrix;
  transformed_data_matrix = gp_exp_quad_cov(d_arr_1, d_sigma, d_len);
  transformed_data_matrix = gp_exp_quad_cov(d_arr_1, d_arr_2, d_sigma, d_len);
  transformed_data_matrix = gp_exp_quad_cov(d_vec_1, d_sigma, d_len);
  transformed_data_matrix = gp_exp_quad_cov(d_vec_1, d_vec_2, d_sigma, d_len);
  transformed_data_matrix = gp_exp_quad_cov(d_rvec_1, d_sigma, d_len);
  transformed_data_matrix = gp_exp_quad_cov(d_rvec_1, d_rvec_2, d_sigma,
                                            d_len);
}
parameters {
  real y_p;
  real p_sigma;
  real p_len;
  array[d_int_1] real p_arr_1;
  array[d_int_2] real p_arr_2;
  array[d_int_1] vector[K] p_vec_1;
  array[d_int_2] vector[K] p_vec_2;
  array[d_int_1] row_vector[K] p_rvec_1;
  array[d_int_2] row_vector[K] p_rvec_2;
}
transformed parameters {
  matrix[d_int_1, d_int_1] transformed_param_matrix;
  transformed_param_matrix = gp_exp_quad_cov(p_arr_1, p_sigma, p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_arr_1, d_sigma, p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_arr_1, p_sigma, d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_arr_1, d_sigma, d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_arr_1, p_arr_2, p_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_arr_1, p_arr_2, d_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_arr_1, p_arr_2, p_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_arr_1, p_arr_2, d_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_arr_1, d_arr_2, p_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_arr_1, d_arr_2, d_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_arr_1, d_arr_2, p_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_arr_1, d_arr_2, d_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(d_arr_1, d_arr_2, p_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(d_arr_1, d_arr_2, d_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(d_arr_1, d_arr_2, p_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(d_arr_1, d_arr_2, d_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(d_arr_1, p_arr_2, p_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(d_arr_1, p_arr_2, d_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(d_arr_1, p_arr_2, p_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(d_arr_1, p_arr_2, d_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_vec_1, p_sigma, p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_vec_1, d_sigma, p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_vec_1, p_sigma, d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_vec_1, d_sigma, d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_vec_1, p_vec_2, p_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_vec_1, p_vec_2, d_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_vec_1, p_vec_2, p_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_vec_1, p_vec_2, d_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_vec_1, d_vec_2, p_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_vec_1, d_vec_2, d_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_vec_1, d_vec_2, p_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_vec_1, d_vec_2, d_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(d_vec_1, d_vec_2, p_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(d_vec_1, d_vec_2, d_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(d_vec_1, d_vec_2, p_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(d_vec_1, p_vec_2, p_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(d_vec_1, p_vec_2, d_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(d_vec_1, p_vec_2, p_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(d_vec_1, p_vec_2, d_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_rvec_1, p_sigma, p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_rvec_1, d_sigma, p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_rvec_1, p_sigma, d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_rvec_1, d_sigma, d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_rvec_1, p_rvec_2, p_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_rvec_1, p_rvec_2, d_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_rvec_1, p_rvec_2, p_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_rvec_1, p_rvec_2, d_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_rvec_1, d_rvec_2, p_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_rvec_1, d_rvec_2, d_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(p_rvec_1, d_rvec_2, p_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(p_rvec_1, d_rvec_2, d_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(d_rvec_1, d_rvec_2, p_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(d_rvec_1, d_rvec_2, d_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(d_rvec_1, d_rvec_2, p_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(d_rvec_1, p_rvec_2, p_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(d_rvec_1, p_rvec_2, d_sigma,
                                             p_len);
  transformed_param_matrix = gp_exp_quad_cov(d_rvec_1, p_rvec_2, p_sigma,
                                             d_len);
  transformed_param_matrix = gp_exp_quad_cov(d_rvec_1, p_rvec_2, d_sigma,
                                             d_len);
}
model {
  y_p ~ normal(0, 1);
}

