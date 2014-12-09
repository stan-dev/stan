parameters {
  real int_x;
  real real_x;
  real vector_x;
  real row_vector_x;
  real matrix_x;
  real unit_vector_x;
  real simplex_x;
  real ordered_x;
  real positive_ordered_x;
  real cholesky_factor_cov_x;
  real cholesky_factor_corr_x;
  real cov_matrix_x;
  real corr_matrix_x;
}
model {
  int_x ~ normal(0,1);
  real_x ~ normal(0,1);
  vector_x ~ normal(0,1);
  row_vector_x ~ normal(0,1);
  matrix_x ~ normal(0,1);
  unit_vector_x ~ normal(0,1);
  simplex_x ~ normal(0,1);
  ordered_x ~ normal(0,1);
  positive_ordered_x ~ normal(0,1);
  cholesky_factor_cov_x ~ normal(0,1);
  cholesky_factor_corr_x ~ normal(0,1);
  cov_matrix_x ~ normal(0,1);
  corr_matrix_x ~ normal(0,1);
}
