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
  real<lower=0> sigma_beta;
  real mu_a;
  real mu_beta;
} 
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] <- beta[1] * x[i] + a[county[i]];
}
model {
  mu_beta ~ normal(0, .0001);
  sigma_beta ~ uniform(0, 100);
  beta ~ normal(mu_beta, sigma_beta);

  mu_a ~ normal(0, .0001);
  sigma_a ~ uniform(0, 100);
  a ~ normal (mu_a, sigma_a);

  sigma_y ~ uniform(0, 100);
  y ~ normal(y_hat, sigma_y);
}
