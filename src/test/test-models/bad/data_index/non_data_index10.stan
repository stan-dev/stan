parameters {
  real y[3];

}
transformed parameters {
  cholesky_factor_cov[size(y)] z;
}
