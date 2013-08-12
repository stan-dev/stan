data {
  int<lower=0> N; 
  int<lower=1,upper=85> county[N];
  vector[N] u;
  vector[N] x;
  vector[N] y;
} 
transformed data {
  vector[N] inter;

  inter <- u .* x;
}
parameters {
  vector[85] a;
  vector[85] b;
  vector[2] beta;
  real mu_a;
  real mu_b;
  real mu_beta;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_b;
  real<lower=0,upper=100> sigma_beta;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- a[county[i]] + x[i] * b[county[i]] + beta[1] * u[i]     
                + beta[2] * inter[i];
}
model {
  mu_beta ~ normal(0, 1);
  beta ~ normal(100 * mu_beta, sigma_beta);

  mu_a ~ normal(0, 1);
  a ~ normal (mu_a, sigma_a);

  mu_b ~ normal(0, 1);
  b ~ normal (0.1 * mu_b, sigma_b);

  y ~ normal(y_hat, sigma_y);
}