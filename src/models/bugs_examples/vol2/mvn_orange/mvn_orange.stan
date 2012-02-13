// Orange Trees (Multi-variate normal) 
// http://www.openbugs.info/Examples/OtreesMVN.html
// and refer to ../orange 

# FIXME: there are some discrepancy for parameters thetavar 
# and the s.e. for some parameters

data {
  int(0,) K;
  int(0,) N;
  int x[N];
  double Y[K, N]; 
  cov_matrix(3) invR;  
  // matrix(3, 3) invR; 
  cov_matrix(3) mu_var_prior; 
  vector(3) mu_m_prior; 
}

parameters{
  double(0,) sigmasq;
  vector(3)  theta[K]; 
  vector(3)  thetamu; 
  cov_matrix(3) thetavar; 
} 

transformed parameters {
  double(0,) sigma; 
  sigma <- sqrt(sigmasq);
} 

model {
  double phi[K, 3]; 
  for (k in 1:K) {
    theta[k] ~ multi_normal(thetamu, thetavar); 
    phi[k, 1] <- exp(theta[k, 1]);
    phi[k, 2] <- exp(theta[k, 2]) - 1;
    phi[k, 3] <- -exp(theta[k, 3]);
  }

  sigmasq ~ inv_gamma(.001, .001); 
  for (k in 1:K) {
    for (n in 1:N)  
      Y[k, n] ~ normal(phi[k, 1] / (1 + phi[k, 2] * exp(phi[k, 3] * x[n])), sigma); 
  }
  thetamu ~ multi_normal(mu_m_prior, mu_var_prior); 
  thetavar ~ inv_wishart(3, invR); 
}

