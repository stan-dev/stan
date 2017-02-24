data {
  real x;
  real y;
}
transformed data {
  vector[2] td_v1 = [ 1, 2]';
  row_vector[2] td_rv1 = [ 1, 2];
  td_rv1 = [ x, y];
  td_rv1 = [ x + y, x - y];
  td_rv1 = [ x^2, y^2];
}
parameters {
  real z;
}
transformed parameters {
  vector[3] tp_v1 = [ 1, 1, 1]';
  row_vector[2] tp_rv1 = [ 1, x];
  tp_rv1 = [ y, z];
}
model {
  z ~ normal(0,1);
}
generated quantities {
  vector[3] gq_v1 = [1, x, y]';
  row_vector[3] gq_rv1 = [1, x, y];
}
