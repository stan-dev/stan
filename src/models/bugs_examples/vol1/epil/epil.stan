# corresponding to JAGS example 
# in R package BUGSExamples 
# [install.packages("BUGSExamples", repos="http://R-Forge.R-project.org")]
# 


data {
  int<lower=0> N; 
  int<lower=0> T; 
  int<lower=0> y[N, T]; 
  int<lower=0> Trt[N]; 
  int<lower=0> V4[T]; 
  real  log_Base4[N];
  real  log_Age[N]; 
  real  BT[N]; 
  real  log_Age_bar; 
  real  Trt_bar; 
  real  BT_bar; 
  real  V4_bar; 
  real  log_Base4_bar; 
} 

parameters {
  real  a0; 
  real  alpha_Base; 
  real  alpha_Trt; 
  real  alpha_BT; 
  real  alpha_Age;
  real  alpha_V4;
  real  b1[N]; 
  real  b[N, T];
  real<lower=0> sigmasq_b; 
  real<lower=0> sigmasq_b1; 
}

transformed parameters {
  real<lower=0> sigma_b; 
  real<lower=0> sigma_b1; 
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

generated quantities {
  real alpha0; 
  # re-calculate intercept on original scale:
  alpha0 <- a0 - alpha_Base * log_Base4_bar - alpha_Trt * Trt_bar
            - alpha_BT * BT_bar - alpha_Age * log_Age_bar - alpha_V4 * V4_bar; 

} 

