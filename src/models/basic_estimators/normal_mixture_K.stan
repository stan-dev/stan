data {
  int(1,) K;           // number of mixture components
  int(1,) N;           // number of data points
  real y[N];           // observations
}
parameters {
  simplex(K) theta;    // mixing proportions
  real mu[K];          // locations of mixture components
  real(0,) sigma[K];   // scales of mixture components
}
model {
  real ps[K];          // temp for log component densities
  for (k in 1:K) {
    mu[k] ~ normal(0,10);
    sigma[k] ~ uniform(0,10);
  }
  for (n in 1:N) {
    for (k in 1:K) {
      ps[k] <- log(theta[k]) 
               + normal_log(y[n],mu[k],sigma[k]);
    }
    lp__ <- lp__ + log_sum_exp(ps);    
  }
}