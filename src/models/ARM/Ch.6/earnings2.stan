data {
  int<lower=0> N; 
  vector[N] earnings;
  vector[N] height;
  vector[N] sex;
} 
transformed data {
  vector[N] log_earnings;
  vector[N] male;

  log_earnings <- log(earnings);
  male <- 2 - sex;
}
parameters {
  vector[3] beta;
  real<lower=0> sigma;
} 
model {
  log_earnings ~ normal(beta[1] + beta[2] * height + beta[3] * male, sigma);
}
