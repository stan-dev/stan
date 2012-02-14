# Sensitivity to prior distributions: application to Magnesium meta-analysis
#  http://www.openbugs.info/Examples/Magnesium.html


# prior specification 3 
# Dumouchel on sigma (??)

data {
  int N; 
  int rc[N]; 
  int rt[N];
  int nc[N]; 
  int nt[N]; 
} 

transformed data {
  real s0_sqrd; 
  s0_sqrd <- 0.1272041; 
} 
#   > print(s0.sqrd);
#   [1] 0.1272041

parameters {
  real(-10, 10) mu;
  real(0, 1) D0; 
  real(0, 1) pc[N]; 
  real theta[N]; 
} 

transformed parameters {
  real pt[N];
  real sigma; 

  sigma <-  sqrt(s0_sqrd) * (1 - D0) / D0; 
 
  for (n in 1:N) 
    pt[n] <- inv_logit(theta[n] + logit(pc[n])); 
    // I.e., logit(pt[n]) - logit(pc[n]) = theta[n] 
} 

model {
  for (n in 1:N) {
    rt[n] ~ binomial(nt[n], pt[n]); 
    rc[n] ~ binomial(nt[n], pc[n]); 
  } 

  # pc ~ uniform(0, 1); // not vectorized? 
  for (n in 1:N) pc[n] ~ uniform(0, 1);

  ## or we can leave out the above line, which contributes
  ## nothing to the posterior

  theta ~ normal(mu, sigma); 
  mu ~ uniform(-10, 10); 

  // prior for D0

} 
