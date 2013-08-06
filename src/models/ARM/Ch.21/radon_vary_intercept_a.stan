data {
  int<lower=0> N; 
  int<lower=0> J; 
  vector[N] y;
  vector[J] u;
  vector[N] x;
  int county[N];
} 
parameters {
  real b;
  vector[J] a;
  vector[2] beta;
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;
  real mu_b;
  real g_0;
  real g_1;
} 
transformed parameters {
  vector[N] y_hat;
  vector[J] a_hat;
  vector[J] e_a;

  for (j in 1:J)
    a_hat[j] <- g_0 + g_1 * u[j];
  e_a <- a - a_hat;

  for (i in 1:N)
    y_hat[i] <- a[county[i]] + x[i] * b;
}
model {
  g_0 ~ normal(0, 100);
  g_1 ~ normal(0, 100);

  sigma_a ~ uniform(0, 100);
  a ~ normal (a_hat, sigma_a);

  mu_b ~ normal(0, 100);
  sigma_b ~ uniform(0, 100);
  b ~ normal (mu_b, sigma_b);

  sigma_y ~ uniform(0, 100);
  y ~ normal(y_hat, sigma_y);
}
