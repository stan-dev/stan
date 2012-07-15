# Dyes: variance components model 
#  http://www.openbugs.info/Examples/Dyes.html

## P.S. How to vectorize y? 
data {
  int BATCHES; 
  int SAMPLES; 
  real y[BATCHES, SAMPLES]; 
  // vector(SAMPLES) y[BATCHES]; 
} 

parameters {
  real(0,) sigmasq_between;
  real(0,) sigmasq_within; 
  real theta;
  real mu[BATCHES]; 
} 

transformed parameters {
  real sigma_between; 
  real sigma_within;
  sigma_between <- sqrt(sigmasq_between); 
  sigma_within <- sqrt(sigmasq_within); 
} 
model {
  theta ~ normal(0.0, 1E5); 
  sigmasq_between ~ inv_gamma(.001, .001); 
  sigmasq_within ~ inv_gamma(.001, .001); 

  mu ~ normal(theta, sigma_between);
  for (n in 1:BATCHES)  
    for (j in 1:SAMPLES) 
      y[n, j] ~ normal(mu[n], sigma_within); 
    // y[n] ~ normal(nu[n], sigma_within); # for vector(SAMPLES) y[BATCHES] ?? 


  ## try different priors 
  // sigmasq_within ~ inv_gamma(2, .01); 
  // sigmasq_between ~ inv_gamma(2, .01); 
   
  ## other parameter of interests 
  // sigmasq_total <- sigmasq_within + sigmasq_between;
  // f_within <- sigmasq_within / sigmasq_total; 
  // f_between <- sigmasq_between / sigmasq_total; 
}

