// Orange Trees (Multi-variate normal) 
// http://www.openbugs.info/Examples/OtreesMVN.html
// refer to ../orange 

data {
  int(0,) K;
  int(0,) N;
  int x[N];
  double Y[K, N]; 
  cov_matrix(3) R;  // R should be positive definite: could cov_matrix be used? 
  // matrix(3, 3) R; 
  cov_matrix(3) mu_var_prior; 
  vector(3) mu_m_prior; 
}

parameters{
  double(0,) sigmasq;
  vector(3)  theta[K]; 
  vector(3)  thetamu; 
  cov_matrix(3) thetavar; 
} 

derived parameters {
  double mu[K, N]; 
  double phi[K, 3]; 
} 

model {
  sigmasq ~ inv_gamma(.001, .001); 
  for (i in 1:K) {
    phi[i, 1] <- exp(theta[i, 1]);
    phi[i, 2] <- exp(theta[i, 2]) - 1;
    phi[i, 3] <- -exp(theta[i, 3]);
    for (n in 1:N) {
      mu[i, n] <- phi[i, 1] / (1 + phi[i, 2] * exp(phi[i, 3] * x[n]));
      Y[i, n] ~ normal(mu[i, n], sqrt(sigmasq)); 
    }
    theta[i] ~ multi_normal(thetamu, thetavar); 
    thetamu ~ multi_normal(mu_m_prior, mu_var_prior); 
    thetavar ~ inv_wishart(R, 3); 
  }
}

