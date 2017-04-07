parameters {
  real y[3];

}
transformed parameters {
  cholesky_factor_corr[size(y)] z;
}
