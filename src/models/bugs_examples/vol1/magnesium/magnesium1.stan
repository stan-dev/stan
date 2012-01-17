# Sensitivity to prior distributions: application to Magnesium meta-analysis
#  http://www.openbugs.info/Examples/Magnesium.html

# prior specification 1 
# see magnesium_all.stan for putting all 6 different 
# prior specification in one posterior 

data {
  int N; 
  int rc[N]; 
  int rt[N];
  int nc[N]; 
  int nt[N]; 
} 

parameters {
  double(-10, 10) mu;
  double(0,) sigmasq; 
  double(0, 1) pc[N]; 
  double theta[N]; 
} 

transformed parameters {
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

  theta ~ normal(mu, sqrt(sigmasq)); 
  mu ~ uniform(-10, 10); 
  // first prior on sigmasq 
  sigmasq ~ inv_gamma(.001, .001); 
} 
