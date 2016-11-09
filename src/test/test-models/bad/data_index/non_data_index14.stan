parameters {
  real y[3];

}
transformed parameters {
  cov_matrix[size(y)] z;
}
