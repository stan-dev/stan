# Sensitivity to prior distributions: application to Magnesium meta-analysis
#  http://www.openbugs.info/Examples/Magnesium.html

# prior specification 1, 2, 3, 4, 5, and 6 combined  
# in one posterior distribution.  
# For individual model specification, see magnesium$i.stan ($i=1,...,6)

data {
  int N; 
  int rc[N]; 
  int rt[N];
  int nc[N]; 
  int nt[N]; 
} 

derived data {
  double s0_sqrd; 
  double psigma; 
  s0_sqrd <- 0.1272041; 
  psigma <- sqrt(s0_sqrd / Phi(.75)); 
} 

parameters {
  double(-10, 10) mu[6];

  double(0,) sigmasq1; 
  double(0, 50) sigmasq2; 
  double(0,)  sigma3; 
  double(0, 1) B04;
  double(0, 1) D05; 
  double(0, )  sigmasq6; 

  double(0, 1) pc[N, 6]; 
  double theta[N, 6]; 
} 

derived parameters {
  double(0, 1) pt[N, 6];
  double yasigma[6]; 
 
  yasigma[1] <- sqrt(sigmasq1); 
  yasigma[2] <- sqrt(sigmasq2); 
  yasigma[3] <- sigma3; 
  yasigma[4] <-  sqrt(s0_sqrd * (1 - B04) / B04); 
  yasigma[5] <-  sqrt(s0_sqrd) * (1 - D05) / D05; 
  yasigma[6] <- sqrt(sigmasq6); 

  for (n in 1:N) 
    for (k in 1:6) 
      pt[n, k] <- inv_logit(theta[n, k] + logit(pc[n, k])); 
    // I.e., logit(pt[n]) - logit(pc[n]) = theta[n] 
} 

model {
  for (n in 1:N) {
    for (k in 1:6) { 
      rt[n] ~ binomial(nt[n], pt[n, k]); 
      rc[n] ~ binomial(nt[n], pc[n, k]); 
    } 
  } 

  # pc ~ uniform(0, 1); // not vectorized? 
  for (n in 1:N) 
    for (k in 1:6) 
      pc[n, k] ~ uniform(0, 1);

  ## or we can leave out the above line, which contributes
  ## nothing to the posterior

  for (k in 1:6) {
    for (n in 1:N)  theta[n, k] ~ normal(mu[k], yasigma[k]);
    mu[k] ~ uniform(-10, 10); 
  }
  // first prior
  sigmasq1 ~ inv_gamma(.001, .001); 
  // second prior
  sigmasq2 ~ uniform(0, 50);
  // third prior
  sigma3 ~ uniform(0, 50);  # could be left out 
  // fourth prior  
  // uniform prior on B04
  // uniform prior on D05
  // Half-normal prior 
  sigmasq6 ~ normal(0, psigma); 

} 
