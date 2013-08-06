data {
  int<lower=0> N; 
  vector[N] y;
  vector[N] x;
  int county[N];
} 
parameters {
  matrix[85,2] eta;
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real mu_a;
}
transformed parameters {
  vector[N] y_hat;
  matrix[85,2] a;

  a <- mu_a + sigma_a * eta;

  for (i in 1:N)
    y_hat[i] <- a[county[i],1] + a[county[i],2] * x[i];
} 
model {
  mu_a ~ normal(0, 100);
  for (j in 1:85)
    eta[j] ~ normal(0, 1);

  y ~ normal(y_hat, sigma_y);
}
