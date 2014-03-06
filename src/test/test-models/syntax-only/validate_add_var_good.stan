data {
  int n;
  real x;
  int nn[3,2];
  real y[5,2];
  vector[3] v;
  row_vector[3] rv;
  simplex[5] sv;
  unit_vector[7] uv;
  ordered[3] ov;
  matrix[4,5] m;
  cov_matrix[3] covm;
  corr_matrix[3] corrm;
}
transformed data {
  int n_td;
  real x_td;
  int nn_td[3,2];
  real y_td[5,2];
  vector[3] v_td;
  row_vector[3] rv_td;
  simplex[5] sv_td;
  unit_vector[7] uv_td;
  ordered[3] ov_td;
  matrix[4,5] m_td;
  cov_matrix[3] covm_td;
  corr_matrix[3] corrm_td;
}
parameters {
  real x_p;
  real y_p[5,2];
  vector[3] v_p;
  row_vector[3] rv_p;
  simplex[5] sv_p;
  unit_vector[7] uv_p;
  ordered[3] ov_p;
  matrix[4,5] m_p;
  cov_matrix[3] covm_p;
  corr_matrix[3] corrm_p;
}
transformed parameters {
  real x_tp;
  real y_tp[5,2];
  vector[3] v_tp;
  row_vector[3] rv_tp;
  simplex[5] sv_tp;
  unit_vector[7] uv_tp;
  ordered[3] ov_tp;
  matrix[4,5] m_tp;
  cov_matrix[3] covm_tp;
  corr_matrix[3] corrm_tp;
}
model {
  int n_l;
  real x_l;
  int nn_l[3,2];
  real y_l[5,2];
  vector[3] v_l;
  row_vector[3] rv_l;
  matrix[4,5] m_l;

  x_p ~ normal(0,1);
}
generated quantities {
  int n_gq;
  real x_gq;
  int nn_gq[3,2];
  real y_gq[5,2];
  vector[3] v_gq;
  row_vector[3] rv_gq;
  simplex[5] sv_gq;
  unit_vector[7] uv_gq;
  ordered[3] ov_gq;
  matrix[4,5] m_gq;
  cov_matrix[3] covm_gq;
  corr_matrix[3] corrm_gq;
}
