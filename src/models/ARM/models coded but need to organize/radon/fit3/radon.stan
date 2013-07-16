data {
  int<lower=0> N; 
  vector[N] y;
  int county[N];
} 
transformed data {
  matrix[N,85] county_factors;
  for (i in 1:N)
    county_factors[i,county[i]] <- 1;
}
parameters {
  vector[85] factor;
  real<lower=0> sigma;
} 
model {
  y ~ normal(county_factors * factor, sigma);
}
