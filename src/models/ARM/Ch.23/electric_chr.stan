data {
  int<lower=0> N;
  int<lower=0> n_pair;
  vector[N] y;
  vector[N] treatment;
  int pair[N];
}
parameters {
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real beta;
  real mu_a;
  vector[n_pair] eta;
}
transformed parameters {
  vector[N] y_hat;
  vector[n_pair] a;

  a <- mu_a + sigma_a * eta;

  for (i in 1:N)
    y_hat[i] <- a[pair[i]] + beta * treatment[i];
}
model {
  mu_a ~ normal(0, 100);
  eta ~ normal(0, 1);
  beta ~ normal(0, 100);
  y ~ normal(y_hat, sigma_y);
}
