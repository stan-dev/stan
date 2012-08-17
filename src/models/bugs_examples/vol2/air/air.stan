data {
  real alpha; 
  real beta; 
  real<lower=0> sigma2; 
  int<lower=0> J; 
  int y[J]; 
  int Z[J]; 
  int n[J]; 
} 

transformed data {
  real<lower=0> sigma; 
  sigma <- sqrt(sigma2); 
} 

parameters {
   real theta1; 
   real theta2; 
   real X[J]; 
} 

model {
  theta1 ~ normal(0, 32);   // 32^2 = 1024 
  theta2 ~ normal(0, 32); 
  for (j in 1:J) {
     X[j] ~ normal(alpha + beta * Z[j], sigma); 
     y[j] ~ binomial(n[j], inv_logit(theta1 + theta2 * X[j]));
  }
}

