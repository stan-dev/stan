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
  vector[85] a;
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real mu_a;
} 
model {
  sigma_y ~ uniform(0, 100);
  mu_a ~ normal(0, .0001);
  sigma_a ~ uniform(0, 100);

  for (j in 1:85)
    a[j] ~ normal (mu_a, sigma_a);
  for (i in 1:N)
    y[i] ~ normal(a[county[i]], sigma_y);
}
