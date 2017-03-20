functions {

  matrix foo() {
    matrix[3,2] result = [ [ 1, 2], [ 3, 4], [5, 6] ];
    return result;
  }

}
transformed data {
  real td_d1 = 1;
  real td_d2 = 2;
  row_vector[2] td_rv2;
  matrix[3,2] td_mat32;
  td_rv2 = [ 10, 100 ];
  td_mat32 = [ [ 1, 2], [ 3, 4], [5, 6] ];
  td_mat32 = [ td_rv2, td_rv2, td_rv2];
  td_mat32 = foo();
}
parameters {
  real p_z;
}
transformed parameters {
  real tp_x;
  real tp_y;
  row_vector[2] tp_rv2;
  matrix[3,2] tp_mat32;
  tp_mat32 = [ td_rv2, td_rv2];
  tp_rv2 = [ td_d1, td_d2];
  tp_mat32 = [ tp_rv2, tp_rv2];
  tp_rv2 = [ tp_x, tp_y];
  tp_mat32 = [ tp_rv2, tp_rv2];
  tp_rv2 = [ td_d1, tp_y];
  tp_mat32 = [ tp_rv2, tp_rv2];
}
model {
  p_z ~ normal(0,1);
}
generated quantities {
  row_vector[2] gq_rv2;
  matrix[3,2] gq_mat32;
  gq_mat32 = [ td_rv2, td_rv2];
  gq_rv2 = [ td_d1, td_d2];
  gq_mat32 = [ gq_rv2, gq_rv2];
  gq_rv2 = [ tp_x, tp_y];
  gq_mat32 = [ gq_rv2, gq_rv2];
  gq_rv2 = [ td_d1, tp_y];
  gq_mat32 = [ gq_rv2, gq_rv2];
}
