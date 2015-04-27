parameters { 
  cov_matrix[3] Sigma;
  vector[3] mu;
  vector[3] sigma;
  real<lower=0> eta;
}
model {
  increment_log_prob(lkj_cov_log(Sigma,mu,sigma,eta));
}
