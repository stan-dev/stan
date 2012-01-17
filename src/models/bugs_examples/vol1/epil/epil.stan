# corresponding to JAGS example 
# in R package BUGSExamples 
# [install.packages("BUGSExamples", repos="http://R-Forge.R-project.org")]
# 


## status: not work (Mon Dec 19 18:08:33 EST 2011)
##         error thrown out 
data {
  int(0,) N; 
  int(0,) T; 
  int(0,) y[N, T]; 
  int(0,) Trt[N]; 
  int(0,) V4[T]; 
  double  log_Base4[N];
  double  log_Age[N]; 
  double  BT[N]; 
  double  log_Age_bar; 
  double  Trt_bar; 
  double  BT_bar; 
  double  V4_bar; 
  double  log_Base4_bar; 
} 

parameters {
  double  a0; 
  double  alpha_Base; 
  double  alpha_Trt; 
  double  alpha_BT; 
  double  alpha_Age;
  double  alpha_V4;
  double  b1[N]; 
  double  b[N, T];
  double(0,) sigmasq_b; 
  double(0,) sigmasq_b1; 
}

transformed parameters {
  double sigma_b; 
  double sigma_b1; 
  sigma_b <- sqrt(sigmasq_b); 
  sigma_b1 <- sqrt(sigmasq_b1); 
} 

model {
  a0 ~ normal(0, 100);
  alpha_Base ~ normal(0, 100);
  alpha_Trt  ~ normal(0, 100);
  alpha_BT   ~ normal(0, 100);
  alpha_Age  ~ normal(0, 100);
  alpha_V4   ~ normal(0, 100);
  sigmasq_b1 ~ inv_gamma(.001, .001);
  sigmasq_b ~ inv_gamma(.001, .001);
  for(n in 1:N) {
    b1[n] ~ normal(0, sigma_b1); 
    for(t in 1:T) {
      b[n, t] ~ normal(0, sigma_b); 
      y[n, t] ~ poisson(exp(a0 + alpha_Base * (log_Base4[n] - log_Base4_bar)   
                            + alpha_Trt * (Trt[n] - Trt_bar)  
                            + alpha_BT  * (BT[n] - BT_bar)  
                            + alpha_Age * (log_Age[n] - log_Age_bar)  
                            + alpha_V4  * (V4[t] - V4_bar) 
                            + b1[n] + b[n, t])); 
    }
  }
}
