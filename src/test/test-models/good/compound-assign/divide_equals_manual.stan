functions {
  vector foo(vector a1) {
    vector[3] lf1;
    int lf_int_a;
    int lf_int_b;
    real lf_real_a;
    real lf_real_b;
    vector[3] lf_v3_b;
    row_vector[3] lf_rv3_b;
    matrix[3, 3] lf_mat33_a;
    matrix[3, 3] lf_mat33_b;
    lf_int_b /= lf_int_a;
    lf_real_b /= lf_int_a;
    lf_real_b /= lf_real_a;
    lf_v3_b /= lf_real_a;
    lf_rv3_b /= lf_real_a;
    lf_mat33_b /= lf_real_a;
    lf1 /= lf_real_a;
    return lf1;
  }
}
data {
  int d_int_a;
  real d_real_a;
  vector[3] d_v3_a;
  row_vector[3] d_rv3_a;
  matrix[3, 3] d_mat33_a;
}
transformed data {
  int td_int_b;
  real td_real_b;
  vector[3] td_v3_b;
  row_vector[3] td_rv3_b;
  matrix[3, 3] td_mat33_b;
  td_int_b /= d_int_a;
  td_real_b /= d_int_a;
  td_real_b /= d_real_a;
  td_v3_b /= d_real_a;
  td_rv3_b /= d_real_a;
  td_mat33_b /= d_real_a;
}
model {
  int l_int_b;
  real l_real_b;
  vector[3] l_v3_b;
  row_vector[3] l_rv3_b;
  matrix[3, 3] l_mat33_b;
  l_int_b /= d_int_a;
  l_real_b /= d_int_a;
  l_real_b /= d_real_a;
  l_v3_b /= d_real_a;
  l_rv3_b /= d_real_a;
  l_mat33_b /= d_real_a;
}
generated quantities {
  int gq_int_b;
  real gq_real_b;
  vector[3] gq_v3_b;
  row_vector[3] gq_rv3_b;
  matrix[3, 3] gq_mat33_b;
  gq_int_b /= d_int_a;
  gq_real_b /= d_int_a;
  gq_real_b /= d_real_a;
  gq_v3_b /= d_real_a;
  gq_rv3_b /= d_real_a;
  gq_mat33_b /= d_real_a;
}
