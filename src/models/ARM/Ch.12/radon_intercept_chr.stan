data {
  int<lower=0> N; 
  vector[N] y;
  int county[N];
} 
parameters {
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real mu_a;
  vector[85] eta;
} 
transformed parameters {
  vector[N] y_hat;
  vector[85] a;

  a <- mu_a + sigma_a * eta;

  for (i in 1:N)
    y_hat[i] <- a[county[i]];
}
model {
  mu_a ~ normal(0, 100);
  eta ~ normal(0, 1);

  y ~ normal(y_hat, sigma_y);
}
