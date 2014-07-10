data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=1,upper=J> county[N];
  vector[J] u;
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[J] a;
  real b;
  vector[2] beta;
  real g_0;
  real g_1;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  vector[J] a_hat;
  vector[J] e_a;
  vector[N] y_hat;

  for (j in 1:J)
    a_hat[j] <- 100 * g_0 + 100 * g_1 * u[j];
  e_a <- a - a_hat;

  for (i in 1:N)
    y_hat[i] <- a[county[i]] + x[i] * b * 100;
}
model {
  g_0 ~ normal(0, 1);
  g_1 ~ normal(0, 1);

  sigma_a ~ uniform(0, 100);
  a ~ normal (a_hat, sigma_a);

  b ~ normal (0, 1);

  sigma_y ~ uniform(0, 100);
  y ~ normal(y_hat, sigma_y);
}
