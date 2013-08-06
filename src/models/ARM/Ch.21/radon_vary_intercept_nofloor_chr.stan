data {
  int<lower=0> N; 
  int<lower=0> J; 
  vector[N] y;
  vector[N] u;
  int county[N];
} 
parameters {
  real b;
  vector[J] eta;
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real mu_a;
} 
transformed parameters {
  vector[N] y_hat;
  vector[J] a;

  a <- mu_a + sigma_a * eta;

  for (i in 1:N)
    y_hat[i] <- a[county[i]] + u[i] * b;
}
model {
  mu_a ~ normal(0, 100);
  eta ~ normal(0, 1);

  b ~ normal(0, 100);

  y ~ normal(y_hat, sigma_y);
}
