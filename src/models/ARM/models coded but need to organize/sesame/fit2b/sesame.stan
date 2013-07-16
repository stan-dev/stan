data {
  int<lower=0> N; 
  vector[N] watched_hat;
  vector[N] y;
}
parameters {
  vector[2] beta;
  real<lower=0> sigma;
} 
model {
  y ~ normal(beta[1] + beta[2] * watched_hat,sigma);
}
