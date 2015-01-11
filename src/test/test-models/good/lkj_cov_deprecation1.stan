parameters { 
  cov_matrix[3] Sigma;
  vector[3] mu;
  vector[3] sigma;
  real<lower=0> eta;
}
model {
  Sigma ~ lkj_cov(mu,sigma,eta);
}
