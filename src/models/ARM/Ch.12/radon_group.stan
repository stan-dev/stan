data {
  int<lower=0> N; 
  vector[N] y;
  vector[N] x;
  vector[N] u;
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
  vector[2] beta;
  real<lower=0> sigma;
} 
model {
  y ~ normal(county_factors * const_coef + beta[1] * x + beta[2] * u, sigma);
}
