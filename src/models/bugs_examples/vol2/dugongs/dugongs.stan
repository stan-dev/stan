// http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol2.pdf
// Page 2: Dugongs 
// 
//
// note that in the original example, the parameter 
// called gamma is called lambda here.  
// 


data {
  int<lower=0> N; 
  real x[N]; 
  real Y[N]; 
} 
parameters {
  real alpha; 
  real beta;  
  real<lower=.5,upper= 1> lambda; // orginal gamma in the JAGS example  
  real<lower=0> tau; 
   
} 
transformed parameters {
  real sigma; 
  real U3; 
  sigma <- 1 / sqrt(tau); 
  U3 <- logit(lambda);
} 
model {
  for (i in 1:N) 
    Y[i] ~ normal(alpha - beta * pow(lambda, x[i]), sigma);
    
  alpha ~ normal(0.0, 1000); 
  beta ~ normal(0.0, 1000); 
  lambda ~ uniform(.5, 1); 
  tau ~ gamma(.0001, .0001); 
}

