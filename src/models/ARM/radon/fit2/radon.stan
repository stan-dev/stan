data {
  int<lower=0> N; 
  vector[N] y;
  vector[N] x;
  int county[N];
} 
transformed data {
  matrix[N,85] county_factors;
  for (i in 1:N)
    county_factors[i,county[i]] <- 1;
}
parameters {
  vector[1] beta;
  vector[85] factor;
  real<lower=0> sigma;
} 
model {
  y ~ normal(beta[1] * x + county_factors * factor, sigma);
}
