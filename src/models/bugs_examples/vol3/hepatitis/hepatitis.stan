
# Hepatitis: a normal hierarchical model with measurement 
# error
#  http://openbugs.info/Examples/Hepatitis.html

# In this version the measurement errors are not modeled.  
# See hepatisisME for the version with measurement errors. 


## note that we have missing data in the orignal data Y[N, T]; 
## here, we turn Y[N, T] into Yvec with the missing
## data removed.  


data {
  int<lower=0> N1;              ## N1 is the length of the vector, Yvec1, that is 
  int<lower=0> N;               ## created from concatenate columns of matrix Y[N, T] 
  real Yvec1[N1];        ## with NA's removed. 
  real tvec1[N1];        ## N is the nrow of original matrix Y[N, T] 
  int<lower=0> idxn1[N1];       ## idxn1 maps Yvec to its orignal n index 
  real y0[N]; 
} 

transformed data {
  real y0_mean; 
  y0_mean <- mean(y0); 
} 

parameters {
  real<lower=0> sigmasq_y; 
  real<lower=0> sigmasq_alpha; 
  real<lower=0> sigmasq_beta; 
  real alpha[N]; 
  real beta[N]; 
  real gamma; 
  real alpha0; 
  real beta0; 
} 

  
transformed parameters {
  real sigma_y; 
  real sigma_alpha; 
  real sigma_beta; 
  sigma_y <- sqrt(sigmasq_y); 
  sigma_alpha <- sqrt(sigmasq_alpha); 
  sigma_beta <- sqrt(sigmasq_beta); 
}
 
model {
  int oldn; 
  for (n in 1:N1) {
    oldn <- idxn1[n]; 
    Yvec1[n] ~ normal(alpha[oldn] + beta[oldn] * (tvec1[n] - 6.5) + gamma * (y0[oldn] - y0_mean), sigma_y); 
  }

  alpha ~ normal(alpha0, sigma_alpha); 
  beta ~ normal(beta0, sigma_beta); 

  sigmasq_y ~ inv_gamma(.001, .001); 
  sigmasq_alpha ~ inv_gamma(.001, .001); 
  sigmasq_beta ~ inv_gamma(.001, .001); 
  alpha0 ~ normal(0, 1000); 
  beta0 ~ normal(0, 1000); 
  gamma ~ normal(0, 1000); 
}
