# http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol1.pdf
# Page 48: Mice Weibull regression 

# not work yet
# note that we have censored data, in which case stan might not support
# need to work around 

# note that stan and JAGS have different parameterization for Weibull
# distribution
 
data {
  int(0,) N; 
  int(0,) M;
  int(0,) group[N];
  int is_censored[N]; 
  int(0,) last_t[N]; 
  double(0,) t[N]; 
}

parameters {
  double beta[M]; 
  double(0,) r; 
} 

// derived parameters {
  // double irr_control; 
  // double veh_control;
  // double test_sub; 
  // double pos_control; 
  // double median[M]; 
// } 
derived parameters {
  double sigma[M]; 
  for (m in 1:M)  sigma[m] <- exp(-beta[m] / r);
} 

model {
  r ~ gamma(1.0, 0.0001);      
  for(j in 1:M) {
    beta[j] ~ normal(0.0, 1E2); 
    // median[j] <- pow(log(2) * exp(-beta[j]), 1 / r);  
  }
  for(i in 1:N) {                          
    // is_censored[i] ~ dinterval(t[i], last_t[i]);
    t[i] ~ weibull(r, sigma[group[i]]); 
  }

  // irr_control <- beta[1];              
  // veh_control <- beta[2] - beta[1]; 
  // test_sub <- beta[3] - beta[1];
  // pos_control <- beta[4] - beta[1];
}

// change the generated cpp code: 
/*
            if (0 == is_censored[i - 1]) 
                lp__ += stan::prob::weibull_log(t[i - 1], r, sigma[group[i - 1] - 1]);
            // right censored: log(1 - F(t_r)), where F: weibull CDF  
            else lp__ += -pow(last_t[i - 1] / sigma[group[i - 1] - 1], r);
**/
