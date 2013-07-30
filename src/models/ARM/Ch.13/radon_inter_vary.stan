data {
  int<lower=0> N; 
  vector[N] y;
  vector[N] x;
  vector[N] u;
  int county[N];
} 
transformed data {
  vector[N] inter;
  inter <- u .* x;
}
parameters {
  vector[85] a;
  vector[85] b;
  vector[2] beta;
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;
  real<lower=0> sigma_beta;
  real mu_a;
  real mu_b;
  real mu_beta;
} 
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] <- a[county[i]] + x[i] * b[county[i]] + beta[1] * u[i] + beta[2] * inter[i];
}
model {
  mu_beta ~ normal(0, .0001);
  sigma_beta ~ uniform(0, 100);
  beta ~ normal(mu_beta, sigma_beta);

  mu_a ~ normal(0, .0001);
  sigma_a ~ uniform(0, 100);
  a ~ normal (mu_a, sigma_a);

  mu_b ~ normal(0, .0001);
  sigma_b ~ uniform(0, 100);
  b ~ normal (mu_b, sigma_b);

  sigma_y ~ uniform(0, 100);
  y ~ normal(y_hat, sigma_y);
}
