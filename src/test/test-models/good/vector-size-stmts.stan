functions {
  void test_vector(int vlen1, int vlen2, int arrlen) {
    vector[vlen1] lf1;
    array[arrlen] vector[vlen2] lf2;
  }
  void test_row_vector(int vlen1, int vlen2, int arrlen) {
    row_vector[vlen1] lf1;
    array[arrlen] row_vector[vlen2] lf2;
  }
  void test_matrix(int nrow1, int ncol1, int nrow2, int ncol2, int arrlen) {
    matrix[nrow1, ncol1] lf1;
    array[arrlen] matrix[nrow2, ncol2] lf2;
  }
}
data {
  int veclen;
  int arrlen;
  int nrows;
  int ncols;
  array[arrlen] real d_real_ar;
  vector[veclen] d_v;
  array[arrlen] vector[veclen] d_v_ar;
  row_vector[veclen] d_rv;
  array[arrlen] row_vector[veclen] d_rv_ar;
  matrix[nrows, ncols] d_m;
  array[arrlen] matrix[nrows, ncols] d_m_ar;
}
transformed data {
  array[arrlen] real td_real_ar = d_real_ar;
  vector[veclen] td_v = d_v;
  array[arrlen] vector[veclen] td_v_ar = d_v_ar;
  row_vector[veclen] td_rv = d_rv;
  array[arrlen] row_vector[veclen] td_rv_ar = d_rv_ar;
  matrix[nrows, ncols] td_m = d_m;
  array[arrlen] matrix[nrows, ncols] td_m_ar = d_m_ar;
  {
    array[arrlen] real local_real_ar = d_real_ar;
    vector[veclen] local_v = d_v;
    array[arrlen] vector[veclen] local_v_ar = d_v_ar;
    row_vector[veclen] local_rv = d_rv;
    array[arrlen] row_vector[veclen] local_rv_ar = d_rv_ar;
    matrix[nrows, ncols] local_m = d_m;
    array[arrlen] matrix[nrows, ncols] local_m_ar = d_m_ar;
  }
}
parameters {
  array[2] real<lower=-10, upper=10> y;
  array[arrlen] real p_real_ar;
  vector[veclen] p_v1;
  array[arrlen] vector[veclen] p_v_ar;
  row_vector[veclen] p_rv;
  array[arrlen] row_vector[veclen] p_rv_ar;
  matrix[nrows, ncols] p_m;
  array[arrlen] matrix[nrows, ncols] p_m_ar;
}
transformed parameters {
  array[arrlen] real tp_real_ar = td_real_ar;
  vector[veclen] tp_v1 = d_v;
  array[arrlen] vector[veclen] tp_v_ar4 = d_v_ar;
  row_vector[veclen] tp_rv = d_rv;
  array[arrlen] row_vector[veclen] tp_rv_ar = d_rv_ar;
  matrix[nrows, ncols] tp_m = d_m;
  array[arrlen] matrix[nrows, ncols] tp_m_ar = d_m_ar;
  {
    array[arrlen] real local2_real_ar = d_real_ar;
    vector[veclen] local2_v = d_v;
    array[arrlen] vector[veclen] local2_v_ar = d_v_ar;
    row_vector[veclen] local2_rv = d_rv;
    array[arrlen] row_vector[veclen] local2_rv_ar = d_rv_ar;
    matrix[nrows, ncols] local2_m = d_m;
    array[arrlen] matrix[nrows, ncols] local2_m_ar = d_m_ar;
  }
}
model {
  array[arrlen] real local3_real_ar = d_real_ar;
  vector[veclen] local3_v = d_v;
  array[arrlen] vector[veclen] local3_v_ar = d_v_ar;
  row_vector[veclen] local3_rv = d_rv;
  array[arrlen] row_vector[veclen] local3_rv_ar = d_rv_ar;
  matrix[nrows, ncols] local3_m = d_m;
  array[arrlen] matrix[nrows, ncols] local3_m_ar = d_m_ar;
  y ~ normal(0, 1);
}
generated quantities {
  array[arrlen] real gq_real_ar = td_real_ar;
  vector[veclen] gq_v1 = d_v;
  array[arrlen] vector[veclen] gq_v_ar4 = d_v_ar;
  row_vector[veclen] gq_rv = d_rv;
  array[arrlen] row_vector[veclen] gq_rv_ar = d_rv_ar;
  matrix[nrows, ncols] gq_m = d_m;
  array[arrlen] matrix[nrows, ncols] gq_m_ar = d_m_ar;
  {
    array[arrlen] real local4_real_ar = d_real_ar;
    vector[veclen] local4_v = d_v;
    array[arrlen] vector[veclen] local4_v_ar = d_v_ar;
    row_vector[veclen] local4_rv = d_rv;
    array[arrlen] row_vector[veclen] local4_rv_ar = d_rv_ar;
    matrix[nrows, ncols] local4_m = d_m;
    array[arrlen] matrix[nrows, ncols] local4_m_ar = d_m_ar;
  }
}

