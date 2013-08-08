data {
  int<lower=0> N; 
  vector[N] y;
  vector[N] x;
  int county[N];
} 
parameters {
  vector[85] a;
  vector[1] beta;
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real mu_a;
} 
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] <- beta[1] * x[i] + a[county[i]];
}
model {
  beta ~ normal(0, 100);
  mu_a ~ normal(0, 100);
  a ~ normal (mu_a, sigma_a);

  y ~ normal(y_hat, sigma_y);
}