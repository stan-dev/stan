# compared with JAGS version in 
# the R package BUGSExamples (https://r-forge.r-project.org/R/?group_id=882) 
data {
  int(0,) Ndogs; 
  int(0,) Ntrials; 
  int Y[Ndogs, Ntrials];
}
transformed data {
  int xa[Ndogs, Ntrials]; 
  int xs[Ndogs, Ntrials]; 
  for (dog in 1:Ndogs) {
    //xa[dog, 1] <- 0; 
    for (trial in 2:Ntrials) {
      //xa[dog, trial] <- sum(Y[dog, 1:(trial-1)]);
      for (k in 1:(trial - 1)) 
        xa[dog, trial] <- xa[dog, trial] + Y[dog, k]; 
      xs[dog, trial] <- trial - 1 - xa[dog, trial];
    }
  } 
} 
parameters {
  real(, -0.00001) alpha;
  real(, -0.00001) beta;
} 
model {
  alpha ~ normal(0.0, 316.2);
  beta  ~ normal(0.0, 316.2);
  for(dog in 1:Ndogs)  
    for (trial in 2:Ntrials)  
      1 - Y[dog, trial] ~ bernoulli(exp(alpha * xa[dog, trial] + beta * xs[dog, trial]));
} 
generated quantities {
  real A;
  real B;
  A <- exp(alpha);
  B <- exp(beta);
}
