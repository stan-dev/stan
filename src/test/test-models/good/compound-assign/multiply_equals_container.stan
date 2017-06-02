functions {
  real foo(real a1) {
    real lf_x;
    lf_x *= a1;
    return lf_x;
  }
  matrix foo_matrix(matrix a1) {
    matrix[2,3] lf_m1;
    lf_m1 *= a1;
    return lf_m1;
  }
}
transformed data {
  real td_x;
  vector[3] td_v1;
  row_vector[3] td_rv1;
  matrix[2,3] td_m1;
  td_v1 *= td_x;
  td_v1 *= foo(td_x);
  td_rv1 *= td_rv1[1];
  td_rv1 *= td_m1;
  td_m1[1] *= td_x;
  td_m1[1] *= td_v1[1];
  td_m1 *= td_m1;
  td_m1 *= foo_matrix(td_m1);
}
transformed parameters {
  vector[3] tp_v1;
  row_vector[3] tp_rv1;
  matrix[2,3] tp_m1;
  tp_v1 *= tp_v1[1];
  tp_rv1 *= tp_rv1[1];
  tp_rv1 *= tp_m1;
  tp_m1 *= tp_m1;
  tp_m1 *= foo_matrix(tp_m1);
  tp_v1 *= td_x;
  tp_v1 *= foo(td_x);
  tp_rv1 *= td_rv1[1];
  tp_rv1 *= td_m1;
  tp_m1[1] *= td_x;
  tp_m1[1] *= td_v1[1];
  tp_m1 *= td_m1;
  tp_m1 *= foo_matrix(td_m1);
}  
generated quantities {
  vector[3] gq_v1;
  row_vector[3] gq_rv1;
  matrix[2,3] gq_m1;
  gq_v1 *= tp_v1[1];
  gq_rv1 *= tp_rv1[1];
  gq_rv1 *= tp_m1;
  gq_rv1 *= td_m1;
  gq_m1 *= tp_m1;
  gq_m1 *= foo_matrix(tp_m1);
}
