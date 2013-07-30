data {
  int<lower=0> N; 
  vector[N] y;
  vector[N] x;
  int county[N];
} 
transformed data {
  matrix[N,85] county_factors;
  for (i in 1:N) {
    county_factors[i,county[i]] <- 1;
  }
}
parameters {
  vector[85] const_coef;
  vector[85] beta;
  real<lower=0> sigma;
} 
model {
  for (n in 1:N)
    y[n] ~ normal(county_factors * const_coef + x[n] * county_factors * beta, sigma);
}
