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
  vector[N] log_Base4;
  vector[N] log_Age;
  vector[N] BT;
  real  log_Age_bar; 
  real  Trt_bar; 
  real  BT_bar; 
  real  V4_bar; 
  real  log_Base4_bar; 
} 

transformed data {
  vector[T] V4_c;
  vector[N] log_Base4_c;
  vector[N] log_Age_c;
  vector[N] BT_c;
  vector[N] Trt_c;
  log_Base4_c <- log_Base4 - log_Base4_bar;
  log_Age_c <- log_Age - log_Age_bar;
  BT_c <- BT - BT_bar;
  for (i in 1:T) 
    V4_c[i] <- V4[i] - V4_bar;
  for (i in 1:N) 
    Trt_c[i] <- Trt[i] - Trt_bar;
} 

parameters {
  real  a0; 
  real  alpha_Base; 
  real  alpha_Trt; 
  real  alpha_BT; 
  real  alpha_Age;
  real  alpha_V4;
  real  b1[N]; 
  vector[T] b[N];
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
  b1 ~ normal(0, sigma_b1); 
  for(n in 1:N) {
    b[n] ~ normal(0, sigma_b); 
    y[n] ~ poisson_log(a0 + alpha_Base * log_Base4_c[n] 
                          + alpha_Trt * Trt_c[n]
                          + alpha_BT  * BT_c[n] 
                          + alpha_Age * log_Age_c[n] 
                          + b1[n] + alpha_V4 * V4_c + b[n]);
  }
}

generated quantities {
  real alpha0; 
  # re-calculate intercept on original scale:
  alpha0 <- a0 - alpha_Base * log_Base4_bar - alpha_Trt * Trt_bar
            - alpha_BT * BT_bar - alpha_Age * log_Age_bar - alpha_V4 * V4_bar; 

} 

