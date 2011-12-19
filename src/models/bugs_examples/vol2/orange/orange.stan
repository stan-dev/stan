// Orange Trees 
// http://www.openbugs.info/Examples/Otrees.html

data {
  int(0,) K;
  int(0,) N;
  int x[N];
  double Y[K, N]; 
}

parameters{
  double(0,) sigmasq;
  double theta[K, 3];
  double theta_mu[3]; 
  double(0,) theta_sigmasq[3]; 
} 

derived parameters {
  double phi[K, 3]; 
  double theta_sigma[3];
  double sigma;
  for (k in 1:K) { 
    phi[k, 1] <- exp(theta[k, 1]);
    phi[k, 2] <- exp(theta[k, 2]) - 1;
    phi[k, 3] <- -exp(theta[k, 3]);
  } 
  for (j in 1:3)
    theta_sigma[j] <- sqrt(theta_sigmasq[j]);
  sigma <- sqrt(sigmasq);
} 

model {
  sigmasq ~ inv_gamma(.001, .001); 
  for (j in 1:3) {
    theta_mu[j] ~ normal(0, 100); 
    theta_sigmasq[j] ~ inv_gamma(.001, .001); 
  }
  for (k in 1:K) {
    for (j in 1:3)
      theta[k, j] ~ normal(theta_mu[j], theta_sigma[j]);
    for (n in 1:N)
      Y[k, n] ~ normal(phi[k,1] / (1 + phi[k,2] * exp(phi[k,3] * x[n])), sigma);
  }
}

