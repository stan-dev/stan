# http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol2.pdf
# Page 29: Ice: non-parametric smoothing in an age-cohort model


data {
  int(0,) N; 
  int(0,) Nage; 
  int(0,) K; 
  int year[N]; 
  int cases[N]; 
  int age[N]; 
  int pyr[N]; 
} 

parameters {
  double alpha[Nage]; 
  double beta[K]; 
  double(0,) sigma; 
} 

// derived parameters {
//   double logRR[K]; 
// } 

model {
  sigma ~ uniform(0, 1); 
  for (k in 1:2)  
     beta[k] ~ normal(0, sigma * 1E3); 
  for (k in 3:K){
     beta[k] ~ normal(2 * beta[k - 1] - beta[k - 2], sigma); 
  } 

  // for (k in 1:K) 
  //   logRR[k] <- beta[k] - beta[5];

  alpha[1] <- 0.0;   
  for (j in 2:Nage) 
     alpha[j - 1] ~ normal(0, 1000); 
   
  for (i in 1:N)  
    cases[i] ~ poisson(exp(log(pyr[i]) + alpha[age[i]] + beta[year[i]]));
} 
