data {
  int<lower=0> N; 
  vector[N] y;
  vector[N] x;
  vector[N] u;
  int county[N];
} 
transformed data {
  vector[N] inter;
  inter <- u .* x;
}
parameters {
  matrix[85,2] eta;
  vector[2] beta;
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real mu_a;
} 
transformed parameters {
  vector[N] y_hat;
  matrix[85,2] a;

  a <- mu_a + sigma_a * eta;

  for (i in 1:N)
    y_hat[i] <- a[county[i],1] + x[i] * a[county[i],2] + beta[1] * u[i] + beta[2] * inter[i];
}
model {
  beta ~ normal(0, 100);

  mu_a ~ normal(0, 100);
  for (j in 1:85)
    eta[j] ~ normal(0, 1);

  y ~ normal(y_hat, sigma_y);
}
