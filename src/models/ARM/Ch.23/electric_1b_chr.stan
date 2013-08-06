data {
  int<lower=0> N;
  int<lower=0> n_pair;
  int pair[N];
  vector[N] y;
  vector[N] treatment;
  vector[N] pre_test;
}
parameters {
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  vector[2] beta;
  real mu_a;
  vector[n_pair] eta;
}
transformed parameters {
  vector[N] y_hat;
  vector[n_pair] a;

  a <- mu_a + sigma_a * eta;

  for (i in 1:N)
    y_hat[i] <- a[pair[i]] + beta[1] * treatment[i] + beta[2] * pre_test[i];
}
model {
  mu_a ~ normal(0, 100);
  beta ~ normal(0, 100);
  eta ~ normal(0, 1);

  y ~ normal(y_hat, sigma_y);
}
