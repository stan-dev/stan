
## estimated the degree of freedom (dof) from 
## samples by sim.stan
# http://www.openbugs.info/Examples/t-df.html

## estimate dof using continuous priors 

data {
  int<lower=0> N; 
  real y[N]; 
} 

parameters {
  // learning about the dof as a continuous quantity
  real<lower=2,upper= 100> d; 
} 

model {
  for (i in 1:N) { 
    y[i] ~ student_t(d, 0, 1); 
  } 
  // d ~ uniform(2, 100); 
} 
