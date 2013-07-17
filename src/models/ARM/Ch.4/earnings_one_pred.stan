data {
  int<lower=0> N; 
  vector[N] earnings;
  vector[N] height;
} 
parameters {
  vector[2] beta;
  real<lower=0> sigma;
} 
model {
  earnings ~ normal(beta[1] + beta[2] * height, sigma);
}
