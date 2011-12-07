# http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol1.pdf
# Page 48: Mice Weibull regression 

# note that stan and JAGS have different parameterization for Weibull
# distribution
 
data {
  int(0,) N_uc; 
  int(0,) N_rc; 
  int(0,) M;
  int(0,) group_uc[N_uc];
  int(0,) group_rc[N_rc];
  int(0,) last_t_rc[N_rc]; 
  double(0,) t_uc[N_uc]; 
}

parameters {
  double beta[M]; 
  double(0,) r; 
} 

derived parameters {
  double sigma[M]; 
  for (m in 1:M)  sigma[m] <- exp(-beta[m] / r);
} 

model {
  r ~ gamma(1.0, 0.0001);      
  for(j in 1:M) {
    beta[j] ~ normal(0.0, 1E2); 
  }
  for(i in 1:N_uc) {                          
    t_uc[i] ~ weibull(r, sigma[group_uc[i]]); 
  }
  for (i in 1:N_rc) {
    1 ~ bernoulli(exp(-pow(last_t_rc[i] / sigma[group_rc[i]], r)));
  } 
} 
