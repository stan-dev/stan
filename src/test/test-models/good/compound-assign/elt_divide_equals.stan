functions {
  matrix foo_matrix(matrix a1, real a2) {
    matrix[2,3] lf_m1 = a1;
    lf_m1 ./= a1;
    return lf_m1;
  }
}
transformed data {
  real td_x;
  vector[3] td_v1;
  row_vector[3] td_rv1;
  matrix[2,3] td_m1;
  td_v1 ./= td_v1;
  td_rv1 ./= td_rv1;
  td_m1 ./= td_m1;
  td_v1 ./= td_x;
  td_rv1 ./= td_x;
  td_m1 ./= td_x;
}
transformed parameters {
  real tp_x;
  vector[3] tp_v1;
  row_vector[3] tp_rv1;
  matrix[2,3] tp_m1;
  tp_v1 ./= tp_v1;
  tp_rv1 ./= tp_rv1;
  tp_m1 ./= tp_m1;
  tp_v1 ./= td_v1;
  tp_rv1 ./= td_rv1;
  tp_m1 ./= td_m1;
  tp_v1 ./= td_x;
  tp_rv1 ./= td_x;
  tp_m1 ./= td_x;
  tp_v1 ./= tp_x;
  tp_rv1 ./= tp_x;
  tp_m1 ./= tp_x;
}  
generated quantities {
  vector[3] gq_v1;
  row_vector[3] gq_rv1;
  matrix[2,3] gq_m1;
  gq_v1 ./= tp_v1;
  gq_rv1 ./= tp_rv1;
  gq_m1 ./= tp_m1;
  gq_v1 ./= td_v1;
  gq_rv1 ./= td_rv1;
  gq_m1 ./= td_m1;
  gq_v1 ./= td_x;
  gq_rv1 ./= td_x;
  gq_m1 ./= td_x;
  gq_v1 ./= tp_x;
  gq_rv1 ./= tp_x;
  gq_m1 ./= tp_x;
}
