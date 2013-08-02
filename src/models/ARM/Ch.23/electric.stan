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
  real<lower=0> sigma_beta;
  real beta;
  real mu_a;
  real mu_beta;
  vector[n_pair] a;
}
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- a[pair[i]] + beta * treatment[i];
}
model {
  mu_a ~ normal(0, .0001);
  sigma_a ~ uniform(0, 100);
  mu_beta ~ normal(0, .0001);
  sigma_beta ~ uniform(0, 100);
  sigma_y ~ uniform(0, 100);

  a ~ normal(mu_a, sigma_a);
  beta ~ normal(mu_beta, sigma_beta);
  y ~ normal(y_hat, sigma_y);
}
