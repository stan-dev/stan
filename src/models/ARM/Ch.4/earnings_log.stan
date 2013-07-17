data {
  int<lower=0> N; 
  vector[N] earnings;
  vector[N] height;
} 
transformed data {
  vector[N] log_earnings;
  log_earnings <- log(earnings);
}
parameters {
  vector[2] beta;
  real<lower=0> sigma;
} 
model {
  log_earnings ~ normal(beta[1] + beta[2] * height, sigma);
}
