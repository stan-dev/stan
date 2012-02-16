# Stacks: robust regression and ridge regression 
#  http://mathstat.helsinki.fi/openbugs/Examples/Stacks.html

# stacks_normal.stan: normal error term 
# stacks_normal2.stan: normal error term with ridge specification on
# coefficients
# ------
# use `make normal' or `make normal2' to build/run the model 

# stacks_dexp.stan: double exponential error term 
# stacks_dexp2.stan: double exponential error term with ridge regression
# specification on coefficients 
# ------
# use `make dexp' or `make dexp2' to build/run the model 

# stacks_t.stan: student T  error term 
# stacks_t2.stan: student T error term with ridge regression specification on
#  coefficients 
# ------
# use `make t' or `make t2' to build/run the model 

data {
  int(0,) N; 
  int(0,) p; 
  real Y[N]; 
  real z[N, p]; 
} 

parameters {
  real beta0; 
  real beta[p]; 
  real(0,) sigmasq; 
} 

transformed parameters {
  real sigma; 
  sigma <- sqrt(sigmasq); 
} 

model {
  # for (n in 1:N) Y[n] ~ normal(beta0 + beta[1] * z[n, 1] + beta[2] * z[n, 2] + beta[3] * z[n, 3], sigma); 
  for (n in 1:N) Y[n] ~ double_exponential(beta0 + beta[1] * z[n, 1] + beta[2] * z[n, 2] + beta[3] * z[n, 3], sigma); 
  beta0 ~ normal(0, 316); 
  beta ~ normal(0, 316); 
  sigmasq ~ inv_gamma(.001, .001); 
} 

