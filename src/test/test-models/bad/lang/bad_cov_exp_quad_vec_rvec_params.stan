data {
  int K;
  int N_1;
  int N_2;
}
parameters {
  real d_sigma;
  real d_len;
  vector[K] d_vec_1[N_1];
  row_vector[K] d_rvec_1[N_2]; // bad mixed Eigen vector types
}
transformed parameters {
  matrix[N_1, N_2] transformed_params_matrix;

  transformed_params_matrix <- cov_exp_qud(d_vec_1, d_rvec_2, d_sigma, d_len);
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
