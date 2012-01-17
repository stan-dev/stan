# Dyes: variance components model 
#  http://www.openbugs.info/Examples/Dyes.html


## status: not working 

## P.S. How to vectorize y? 

data {
  int BATCHES; 
  int SAMPLES; 
  double y[BATCHES, SAMPLES]; 

  // vector(SAMPLES) y[BATCHES]; 
} 

parameters {
  double theta;
  double(0,) sigmasq_within; 
  double(0,) sigmasq_between;
  double mu[BATCHES]; 
} 

transformed parameters {
  double sigma_within;
  double sigma_between; 
  sigma_between <- sqrt(sigmasq_between); 
  sigma_within <- sqrt(sigmasq_within); 
} 
model {
  mu ~ normal(theta, sigma_between); 
  for (n in 1:BATCHES)  
    for (j in 1:SAMPLES) y[n, j] ~ normal(mu[n], sigma_within); 
    // y[n] ~ normal(nu[n], sigma_within); # for vector(SAMPLES) y[BATCHES] ?? 

  theta ~ normal(0.0, 3.2); 
  sigmasq_between ~ inv_gamma(.001, .001); 
  sigmasq_within ~ inv_gamma(.001, .001); 

  ## try different priors 
  // sigmasq_within ~ inv_gamma(2, .01); 
  // sigmasq_between ~ inv_gamma(2, .01); 
   
  ## other parameter of interests 
  // sigmasq_total <- sigmasq_within + sigmasq_between;
  // f_within <- sigmasq_within / sigmasq_total; 
  // f_between <- sigmasq_between / sigmasq_total; 
}

