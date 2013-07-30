data {
  int<lower=0> N; 
  vector[N] y;
  int county[N];
} 
parameters {
  vector[85] a;
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real mu_a;
} 
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] <- a[county[i]];
}
model {
  mu_a ~ normal(0, .0001);
  sigma_y ~ uniform(0, 100);
  sigma_a ~ uniform(0, 100);
  a ~ normal (mu_a, sigma_a);
  y[i] ~ normal(y_hat, sigma_y);
}
