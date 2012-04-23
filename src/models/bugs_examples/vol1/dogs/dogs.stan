
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
  for (i in 1:Ndogs) {
    // xa[i, 1] <- 0; 
    for (j in 2 : Ntrials) {
      xs[i, j] <- 0; 
      for (k in 1:(j - 1)) xa[i, j] <- xa[i, j] + Y[i, k]; 
      // xa[i, j] <- sum(Y[i, 1:(j - 1)])
      xs[i, j] <- j - 1 - xa[i, j];
    }
  } 
} 

parameters {
  real(, -0.00001) alpha;
  real(, -0.00001) beta;
} 

model {
  // alpha ~ normal_trunc_h_propto(0, 316, -0.00001); 
  // beta ~ normal_trunc_h_propto(0, 316, -0.00001); 
  // alpha ~ normal_trunc_h(0, 316, -0.00001); 
  // beta ~ normal_trunc_h(0, 316, -0.00001); 
  alpha ~ normal(0.0, 316.0); 
  beta  ~ normal(0.0, 316.0); 
  for(i in 1:Ndogs)  
    for (j in 2:Ntrials)  
      1 - Y[i, j] ~ bernoulli(exp(alpha * xa[i, j] + beta * xs[i, j]));
} 
