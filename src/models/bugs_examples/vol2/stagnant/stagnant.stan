# Stagnant: a changepoint problem and an illustration of how NOT to do MCMC!
# 
# Change point model with very poor parameterization stated in WINBUGS examples

data {
  int(0,) N; 
  double x[N]; 
  double Y[N]; 
  
} 

derived data {
  double punif[N]; 
  // matrix(N, 1) punif; ?? 
  for (i in 1:N)  punif[i] <- 1.0 / N; 
} 

parameters {
  double alpha;
  double beta[2]; 
  double(0,) tau; 
  int(0,) k;
} 

derived parameters {
  double(0,) sigma;
  sigma <- 1 / sqrt(tau); 
} 

model {
  k ~ categorical(punif); 
  tau ~ gamma(.001, .001);
  alpha ~ normal(0, 1000); 
  for (i in 1:2) beta[i] ~ normal(0, 1000);
  for (i in 1:N)  
    Y[i] ~ normal(alpha + beta[1 + step(i - k - .5)] * (x[i] - x[k]), sigma); 
} 

