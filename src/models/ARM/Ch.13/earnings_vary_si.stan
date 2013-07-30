data {
  int<lower=0> N; 
  vector[N] earn;
  vector[N] height;
  int eth[N];
} 
transformed data {
  matrix[N,4] ethn_factors;
  vector[N] log_earn;
  log_earn <- log(earn);
  for (i in 1:N) {
    ethn_factors[i,county[i]] <- 1;
  }
}
parameters {
  vector[4] const_coef;
  vector[4] beta;
  real<lower=0> sigma;
} 
model {
  for (n in 1:N)
    log_earn[n] ~ normal(ethn_factors * const_coef + height[n] * ethn_factors * beta, sigma);
}
