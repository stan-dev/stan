data {
  int<lower=0> K;
  int<lower=0> T;
  real y[T];
}

parameters {
  real alpha;
  real beta[K];
  real<lower=0> sigma;
}

model {
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ cauchy(0, 2.5);
  
  for (t in (K+1):T) {
    real mu;
    mu = alpha;
    
    for (k in 1:K)
      mu = mu + beta[k] * y[t - k];
    
    y[t] ~ normal(mu, sigma);
  }
}
