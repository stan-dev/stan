data {
  int<lower=0> N;
  int<lower=0> n_pair;
  int<lower=1,upper=n_pair> pair[N];
  vector[N] treatment;
  vector[N] y;
}
parameters {
  vector[n_pair] a;
  real beta;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
}
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- a[pair[i]] + beta * treatment[i];
}
model {
  mu_a ~ normal(0, 1);
  sigma_a ~ uniform(0, 100);
  sigma_y ~ uniform(0, 100);

  a ~ normal(100 * mu_a, sigma_a);
  beta ~ normal(0, 1);
  y ~ normal(y_hat, sigma_y);
}
