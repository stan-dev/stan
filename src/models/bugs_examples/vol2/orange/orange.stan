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
  double mu[K, N]; 
  double phi[K, 3]; 
  for (i in 1:K) { 
    phi[i, 1] <- exp(theta[i, 1]);
    phi[i, 2] <- exp(theta[i, 2]) - 1;
    phi[i, 3] <- -exp(theta[i, 3]);
  } 
} 

model {
  sigmasq ~ inv_gamma(.001, .001); 
  for (j in 1:3) {
    theta_mu[j] ~ normal(0, 100); 
    theta_sigmasq[j] ~ inv_gamma(.001, .001); 
  }
  for (i in 1:K) {
    for (j in 1:3) {
      theta[i, j] ~ normal(theta_mu[j], sqrt(theta_sigmasq[j]));
    }
    for (n in 1:N) {
      mu[i, n] <- phi[i, 1] / (1 + phi[i, 2] * exp(phi[i, 3] * x[n]));
      Y[i, n] ~ normal(mu[i, n], sqrt(sigmasq)); 
    }
  }
}

