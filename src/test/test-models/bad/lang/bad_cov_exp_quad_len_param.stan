data {
  int K;
  int N_1;
  int N_2;
  vector[K] d_vec_1[N_1];
  vector[K] d_rvec_1[N_2];
}
parameters {
  real d_len[K]; // bad d_len type
  real d_sigma;
}
transformed parameters {
  matrix[N_1, N_2] transformed_param_matrix;

  transformed_param_matrix <- cov_exp_qud(d_vec_1, d_rvec_2, d_sigma, d_len);
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
