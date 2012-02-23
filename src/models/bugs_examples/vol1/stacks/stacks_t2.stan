# Stacks: robust regression and ridge regression 
#  http://mathstat.helsinki.fi/openbugs/Examples/Stacks.html

# stacks_normal.stan: normal error term 
# stacks_normal2.stan: normal error term with ridge specification on
# coefficients
# ------
# use `make normal' or `make normal2' to build/run the model 

# stacks_dexp.stan: real exponential error term 
# stacks_dexp2.stan: real exponential error term with ridge regression
# specification on coefficients 
# ------
# use `make dexp' or `make dexp2' to build/run the model 

# stacks_t.stan: student T  error term 
# stacks_t2.stan: student T error term with ridge regression specification on
#  coefficients 
# ------
# use `make t' or `make t2' to build/run the model 
# 

data {
  int(0,) N; 
  int(0,) p; 
  real Y[N]; 
  matrix(N,p) x; 
} 

// to standardize the x's 
transformed data {
  real z[N, p]; 
  for (j in 1:p) { 
    real mean_x; 
    real sd_x; 
    mean_x <- mean(col(x, j)); 
    sd_x <- sd(col(x, j)); 
    for (i in 1:N)  z[i, j] <- (x[i, j] - mean_x) / sd_x; 
  } 
} 


parameters {
  real beta0; 
  real beta[p]; 
  real(0,) sigmasq; 
  real(0,) sigmasq_beta;
} 

transformed parameters {
  real sigma; 
  real sigma_beta; 
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

