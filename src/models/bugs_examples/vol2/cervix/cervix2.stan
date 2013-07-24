# Cervix: case - control study with errors in covariates 
#  http://www.openbugs.info/Examples/Cervix.html

# from the JAGS readme in classic-bugs/vol2/cervix:

## ``The epidemiology in this example is a little out of date. It is now known
##   that Human Papillomavirus (HPV) is a necessary cause of cervical cancer.
##   Although HSV-2 may have a role as a cofactor in some cases, trying to
##   model its effect on cervical cancer without taking into account HPV is
##   rather pointless.'' 

## In this version, the binary missing x's are integrated out. 

data {
  int<lower=0> Nc; 
  int<lower=0> Ni; 
  int xc[Nc];
  int wc[Nc];
  int dc[Nc];
  int wi[Ni];
  int di[Ni];
} 

parameters {
  real<lower=0,upper=1> phi[2, 2];
  real<lower=0,upper=1> q; 
  real beta0C; 
  real beta; 
  # note that xi is discrete parameters with support {0, 1} 
  # integrated out here 
  //int xi[Ni]; 
} 

model {
  for (n in 1:Nc) {
    xc[n] ~ bernoulli(q); 
    dc[n] ~ bernoulli_logit(beta0C + beta * xc[n]); 
    wc[n] ~ bernoulli(phi[xc[n] + 1, dc[n] + 1]); 
  } 
  for (n in 1:Ni) {
    // xi[n] ~ bernoulli(q); 
    di[n] ~ bernoulli(inv_logit(beta0C + beta) * q + inv_logit(beta0C) * (1 - q)); 
    wi[n] ~ bernoulli(phi[1, di[n] + 1] * (1 - q) + phi[2, di[n] + 1] * q); 
  } 
  q ~ uniform(0, 1); 
  beta0C ~ normal(0, 320); 
  beta ~ normal(0, 320);
  for (i in 1:2) 
    for (j in 1:2) 
      phi[i, j] ~ uniform(0, 1); 
} 

generated quantities {
  real gamma1; 
  real gamma2; 
  # calculate gamma1 = P(x=1|d=0) and gamma2 = P(x=1|d=1) 
  gamma1 <- 1 / (1 + (1 + exp(beta0C + beta)) / (1 + exp(beta0C)) * (1 - q) / q); 
  gamma2 <- 1 / (1 + (1 + exp(-beta0C - beta)) / (1 + exp(-beta0C)) * (1 - q) / q);
} 


