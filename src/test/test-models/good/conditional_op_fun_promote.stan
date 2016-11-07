functions {
  matrix foo(matrix mat, int idx) {
    matrix[rows(mat),cols(mat)] out;
    out[idx,idx] = mat[idx,idx] * (idx==1 ? 0 : mat[idx,idx]);
    return out;
  }
}
parameters {
  real y;
}
transformed parameters{
  matrix[3,3] tp_mat;
  matrix[3,3] tp_mat_foo;
  tp_mat_foo = foo(tp_mat,1);
}
model {
  y ~ normal(0,1);
}
