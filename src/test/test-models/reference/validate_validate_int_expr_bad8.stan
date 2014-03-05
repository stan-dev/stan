data {
  vector[3] a;
  row_vector[3] b;
  matrix[3,3] c;
  unit_vector[3] d;
  simplex[3] e;
  ordered[3] f;
  positive_ordered[3.2] g;
  cholesky_factor_cov[4,5] h;
  cholesky_factor_cov[3] j;
  cov_matrix[3] k;
  corr_matrix[3] l;
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
