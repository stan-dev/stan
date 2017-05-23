functions {

  vector foo(int d) {
    vector[3] result = [10.1, 11*3.0, d]';
    return result;
  }

  row_vector bar() {
    row_vector[2] result = [7, 8];
    return result;
  }

}
data {
  real x;
  real y;
}
transformed data {
  vector[3] td_v1 = [ 21, 22, 23]';
  row_vector[2] td_rv1 = [ 1, 2];
  td_rv1 = [ x, y];
  td_rv1 = [ x + y, x - y];
  td_rv1 = [ x^2, y^2];
  td_v1 = foo(1);
  td_rv1 = bar();
}
parameters {
  real z;
}
transformed parameters {
  vector[3] tp_v1 = [ 41, 42, 43]';
  row_vector[2] tp_rv1 = [ 1, x];
  tp_v1 = foo(1);
  tp_v1 = [ 51, y, z]';
  tp_rv1 = [ y, z];
  tp_rv1 = bar();
}
model {
  z ~ normal(0,1);
}
generated quantities {
  vector[3] gq_v1 = [1, x, y]';
  row_vector[3] gq_rv1 = [1, x, y];
  row_vector[3] gq_rv2 = [1, x, z];
  gq_v1 = foo(1);
}
