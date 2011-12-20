# Stacks: robust regression and ridge regression 
#  http://mathstat.helsinki.fi/openbugs/Examples/Stacks.html

# stacks_normal.stan: normal error term 
# stacks_normal2.stan: normal error term with ridge specification on
# coefficients
# ------
# use `make normal' or `make normal2' to build/run the model 

# stacks_dexp.stan: double exponential error term 
# stacks_dexp2.stan: double exponential error term with regression
# specification on coefficients 
# ------
# use `make dexp' or `make dexp2' to build/run the model 

# stacks_t.stan: student T  error term 
# stacks_t2.stan: student T error term with regression specification on
#  coefficients 
# ------
# use `make t' or `make t2' to build/run the model 
# 
data {
  int(0,) N; 
  int(0,) p; 
  double Y[N]; 
  double z[N, p]; 
} 

parameters {
  double beta0; 
  double beta[p]; 
  double(0,) sigmasq; 
  double(0,) sigmasq_beta;
} 

derived parameters {
  double(0,) sigma; 
  double(0,) sigma_beta; 
  sigma <- sqrt(sigmasq); 
  sigma_beta <- sqrt(sigmasq_beta); 
} 

model {
  # for (n in 1:N) Y[n] ~ normal(beta0 + beta[1] * z[n, 1] + beta[2] * z[n, 2] + beta[3] * z[n, 3], sigma); 
  for (n in 1:N) Y[n] ~ student_t(4, beta0 + beta[1] * z[n, 1] + beta[2] * z[n, 2] + beta[3] * z[n, 3], sigma); 
  beta0 ~ normal(0, 316); 
  # beta ~ normal(0, 316); 
  beta ~ normal(0, sigma_beta); 
  sigmasq ~ inv_gamma(.001, .001); 
  sigmasq_beta ~ inv_gamma(.001, .001); 
} 

