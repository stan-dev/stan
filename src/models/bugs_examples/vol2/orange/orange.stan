// Orange Trees 
// http://www.openbugs.info/Examples/Otrees.html

# status: error thrown out during execution immediately 

data {
  int<lower=0> K;
  int<lower=0> N;
  int x[N];
  real Y[K, N]; 
}

parameters{
  real<lower=0> tau_C;
  real theta[K, 3];
  real mu[3]; 
  real<lower=0> tau[3];
} 

transformed parameters {
  real phi[K, 3]; 
  real sigma[3];
  real sigma_C;
  for (k in 1:K) { 
    phi[k, 1] <- exp(theta[k, 1]);
    phi[k, 2] <- exp(theta[k, 2]) - 1;
    phi[k, 3] <- -exp(theta[k, 3]);
  } 
  for (j in 1:3)
    sigma[j] <- 1 / sqrt(tau[j]);
  sigma_C <- 1 / sqrt(tau_C);
} 

model {
  tau_C ~ gamma(0.001, 0.001); 
  mu ~ normal(0, 100); 
  for (j in 1:3) {
    tau[j] ~ gamma(.001, .001); 
  }
  for (k in 1:K) {
    for (j in 1:3)
      theta[k, j] ~ normal(mu[j], sigma[j]);
    for (n in 1:N)
      Y[k, n] ~ normal(phi[k,1] / (1 + phi[k,2] * exp(phi[k,3] * x[n])), sigma_C);
  }
}
