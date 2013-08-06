data {
  int<lower=0> N;
  int<lower=0> J;
  int county[N];
  vector[N] y;
}
parameters {
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real mu_a;
  vector[J] eta;
}
transformed parameters {
  vector[N] y_hat;
  vector[J] a;
  vector[N] e_y;
  real<lower=0> s_y;
  real<lower=0> s_a;

  a <- mu_a + sigma_a * eta;

  for (i in 1:N)
    y_hat[i] <- a[county[i]];

  e_y <- y - y_hat;
  s_a <- sd(a);
  s_y <- sd(e_y);
}
model {
  mu_a ~ normal(0, 100);

  eta ~ normal(0, 1);
  y ~ normal(y_hat, sigma_y);
}
