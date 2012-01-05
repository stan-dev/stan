# Sensitivity to prior distributions: application to Magnesium meta-analysis
#  http://www.openbugs.info/Examples/Magnesium.html


# prior specification 3 
# uniform on sigma 

data {
  int N; 
  int rc[N]; 
  int rt[N];
  int nc[N]; 
  int nt[N]; 
} 

parameters {
  double(-10, 10) mu;
  double(0, 50) sigma; 
  double(0, 1) pc[N]; 
  double theta[N]; 
} 

derived parameters {
  double pt[N];
 
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
  sigma ~ uniform(0, 50);  # could be left out 
} 
