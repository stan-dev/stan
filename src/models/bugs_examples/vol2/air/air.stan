data {
  real alpha; 
  real beta; 
  real<lower=0> sigma2; 
  int<lower=0> J; 
  int y[J]; 
  vector[J] Z;
  int n[J]; 
} 

transformed data {
  real<lower=0> sigma; 
  sigma <- sqrt(sigma2); 
} 

parameters {
   real theta1; 
   real theta2; 
   vector[J] X; 
} 

model {
  real p[J];
  theta1 ~ normal(0, 32);   // 32^2 = 1024 
  theta2 ~ normal(0, 32); 
  X ~ normal(alpha + beta * Z, sigma);
  y ~ binomial_logit(n, theta1 + theta2 * X);
}

