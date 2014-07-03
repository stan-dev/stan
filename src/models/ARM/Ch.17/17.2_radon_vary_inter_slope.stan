data {
  int<lower=0> N;
  int<lower=0> J;
  vector[N] y;
  vector[N] u;
  int<lower=0,upper=1> x[N];
  int county[N];
}
parameters {
  real<lower=0> sigma;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;
  real g_a_0;
  real g_a_1;
  real g_b_0;
  real g_b_1;
  vector[J] a;
  vector[J] b;
  real mu_a;
  real mu_b;
}
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- a[county[i]] + b[county[i]] * x[i];
}
model {
  vector[N] a_hat;
  vector[N] b_hat;

  mu_a ~ normal(0, 100);
  mu_b ~ normal(0, 100);

  a ~ normal(mu_a, sigma_a);
  b ~ normal(mu_b, sigma_b);

  g_a_0 ~ normal(0, 100);
  g_a_1 ~ normal(0, 100);
  g_b_0 ~ normal(0, 100);
  g_b_1 ~ normal(0, 100);

  a_hat <- g_a_0 + g_a_1 * u;
  b_hat <- g_b_0 + g_b_1 * u;

  y ~ normal(y_hat, sigma);
}
