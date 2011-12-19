# 
# http://www.openbugs.info/Examples/Blockers.html

# status (works: fairly close to results from package BUGSExampels) 

data {
  int(0,) N; 
  int(0,) nt[N]; 
  int(0,) rt[N]; 
  int(0,) nc[N]; 
  int(0,) rc[N]; 
} 
parameters {
  double d; 
  double(0,) sigmasq_delta; 
  double mu[N]; 
  double delta[N]; 
} 

derived parameters {
  double(0,) sigma_delta; 
  sigma_delta <- sqrt(sigmasq_delta); 
} 

model {
  for (n in 1:N) {
    rt[n] ~ binomial(nt[n], inv_logit(mu[n] + delta[n])); 
    rc[n] ~ binomial(nc[n], inv_logit(mu[n]));
    delta[n] ~ student_t(4, d, sigma_delta); 
    mu[n] ~ normal(0.0, 316); # 316^2 = 1E5 
  }
  d ~ normal(0.0, 1.0E3); 
  sigmasq_delta ~ inv_gamma(.001, .001); 

  // do not think stan supports predictive posterior
  // delta_new ~ student_t(4, d, sigma_delta); 
}
