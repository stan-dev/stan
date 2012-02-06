# Jaws: repeated measures analysis of variance
#  http://www.openbugs.info/Examples/Jaws.html

data {
  int(0,) N; 
  int(0,) M; 
  vector(M) Y[N]; 
  double age[M]; 
  cov_matrix(M) R; 
} 

transformed data {
  double mean_age;
  mean_age <- mean(age); 
} 


parameters {
  double beta0; 
  double beta1; 
  cov_matrix(M) Sigma; 
} 

transformed parameters {
  vector(M) mu;
  // for (m in 1:M) mu[m] <- beta0 + beta1 * (age[m] - mean_age); 
 
  for (m in 1:M)  mu[m] <- beta0 + beta1 * age[m]; 
}  

model {
  for (n in 1:N) Y[n] ~ multi_normal(mu, Sigma); 
  Sigma ~ inv_wishart(4, R); 
  beta0 ~ normal(0, 32);
  beta1 ~ normal(0, 32);
} 
