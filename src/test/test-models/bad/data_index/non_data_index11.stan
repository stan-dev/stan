parameters {
  real y[3];

}
transformed parameters {
  cholesky_factor_cov[3, size(y)] z;
}
