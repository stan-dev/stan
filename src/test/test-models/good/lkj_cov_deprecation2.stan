parameters {
  cov_matrix[3] Sigma;
  vector[3] mu;
  vector[3] sigma;
  real<lower=0> eta;
}
model {
  target += lkj_cov_log(Sigma, mu, sigma, eta);
}

