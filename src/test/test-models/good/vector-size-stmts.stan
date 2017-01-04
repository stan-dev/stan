functions {
  void test_vector(int vlen1, int vlen2, int arrlen) {
    vector[vlen1] lf1;
    vector[vlen2] lf2[arrlen];
  }
  void test_row_vector(int vlen1, int vlen2, int arrlen) {
    row_vector[vlen1] lf1;
    row_vector[vlen2] lf2[arrlen];
  }
  void test_matrix(int nrow1, int ncol1, int nrow2, int ncol2, int arrlen) {
    matrix[nrow1,ncol1] lf1;
    matrix[nrow2,ncol2] lf2[arrlen];
  }
}
data {
  int veclen;
  int arrlen;
  int nrows;
  int ncols;
  real d_real_ar[arrlen];
  vector[veclen] d_v;
  vector[veclen] d_v_ar[arrlen];
  row_vector[veclen] d_rv;
  row_vector[veclen] d_rv_ar[arrlen];
  matrix[nrows, ncols] d_m;
  matrix[nrows, ncols] d_m_ar[arrlen];
}
transformed data {
  real td_real_ar[arrlen] = d_real_ar;
  vector[veclen] td_v = d_v;
  vector[veclen] td_v_ar[arrlen] = d_v_ar;
  row_vector[veclen] td_rv = d_rv;
  row_vector[veclen] td_rv_ar[arrlen] = d_rv_ar;
  matrix[nrows, ncols] td_m = d_m;
  matrix[nrows, ncols] td_m_ar[arrlen] = d_m_ar;
  {
    real local_real_ar[arrlen] = d_real_ar;
    vector[veclen] local_v = d_v;
    vector[veclen] local_v_ar[arrlen] = d_v_ar;
    row_vector[veclen] local_rv = d_rv;
    row_vector[veclen] local_rv_ar[arrlen] = d_rv_ar;
    matrix[nrows, ncols] local_m = d_m;
    matrix[nrows, ncols] local_m_ar[arrlen] = d_m_ar;
  }
}
parameters {
  real<lower=-10, upper=10> y[2];

  real p_real_ar[arrlen];
  vector[veclen] p_v1;
  vector[veclen] p_v_ar[arrlen];
  row_vector[veclen] p_rv;
  row_vector[veclen] p_rv_ar[arrlen];
  matrix[nrows, ncols] p_m;
  matrix[nrows, ncols] p_m_ar[arrlen];
}
transformed parameters {
  real tp_real_ar[arrlen] = td_real_ar;
  vector[veclen] tp_v1 = d_v;
  vector[veclen] tp_v_ar4[arrlen] = d_v_ar;
  row_vector[veclen] tp_rv = d_rv;
  row_vector[veclen] tp_rv_ar[arrlen] = d_rv_ar;
  matrix[nrows, ncols] tp_m = d_m;
  matrix[nrows, ncols] tp_m_ar[arrlen] = d_m_ar;
  {
    real local2_real_ar[arrlen] = d_real_ar;
    vector[veclen] local2_v = d_v;
    vector[veclen] local2_v_ar[arrlen] = d_v_ar;
    row_vector[veclen] local2_rv = d_rv;
    row_vector[veclen] local2_rv_ar[arrlen] = d_rv_ar;
    matrix[nrows, ncols] local2_m = d_m;
    matrix[nrows, ncols] local2_m_ar[arrlen] = d_m_ar;
  }
}
model {
  real local3_real_ar[arrlen] = d_real_ar;
  vector[veclen] local3_v = d_v;
  vector[veclen] local3_v_ar[arrlen] = d_v_ar;
  row_vector[veclen] local3_rv = d_rv;
  row_vector[veclen] local3_rv_ar[arrlen] = d_rv_ar;
  matrix[nrows, ncols] local3_m = d_m;
  matrix[nrows, ncols] local3_m_ar[arrlen] = d_m_ar;

  y ~ normal(0,1);
}
generated quantities {
  real gq_real_ar[arrlen] = td_real_ar;
  vector[veclen] gq_v1 = d_v;
  vector[veclen] gq_v_ar4[arrlen] = d_v_ar;
  row_vector[veclen] gq_rv = d_rv;
  row_vector[veclen] gq_rv_ar[arrlen] = d_rv_ar;
  matrix[nrows, ncols] gq_m = d_m;
  matrix[nrows, ncols] gq_m_ar[arrlen] = d_m_ar;
  {
    real local4_real_ar[arrlen] = d_real_ar;
    vector[veclen] local4_v = d_v;
    vector[veclen] local4_v_ar[arrlen] = d_v_ar;
    row_vector[veclen] local4_rv = d_rv;
    row_vector[veclen] local4_rv_ar[arrlen] = d_rv_ar;
    matrix[nrows, ncols] local4_m = d_m;
    matrix[nrows, ncols] local4_m_ar[arrlen] = d_m_ar;
  }
}
