# http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol2.pdf
# Page 29: Ice: non-parametric smoothing in an age-cohort model

// The model is the same specified here as in JAGS example, 
// but this example's JAGS version is different from that of 
// WinBUGS because JAGS does not support some feature. 
// TODO: 
// Maybe we should check if we could specify 
// the model the same as WinBUGS? 

// status: the results are farely cose to those 
// from JAGS 
data {
  int<lower=0> N; 
  int<lower=0> Nage; 
  int<lower=0> K; 
  int year[N]; 
  int cases[N]; 
  int age[N]; 
  int pyr[N]; 
  real alpha1; 
} 

parameters {
  real alpha[Nage - 1]; 
  real beta[K]; 
  real<lower=0,upper=1> sigma; 
} 

model {
  vector[N] r;

  sigma ~ uniform(0, 1); 
  for (k in 1:2)  beta[k] ~ normal(0, sigma * 1E3); 
  for (k in 3:K)  beta[k] ~ normal(2 * beta[k - 1] - beta[k - 2], sigma); 
  alpha ~ normal(0, 1000); 

  for (i in 1:N) { 
    if (age[i] == 1) r[i] <- alpha1 + log(pyr[i]) + beta[year[i]];
    else r[i] <- alpha[age[i] - 1] + log(pyr[i]) + beta[year[i]];
  } 

  cases ~ poisson_log(r);
} 

generated quantities {
  real logRR[K];
  for (k in 1:K) 
    logRR[k] <- beta[k] - beta[5];
}
