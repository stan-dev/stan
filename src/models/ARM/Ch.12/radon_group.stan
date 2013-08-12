data {
  int<lower=0> N; 
  int<lower=1,upper=85> county[N];
  vector[N] u;
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[85] b;
  vector[2] beta;
  real mu_b;
  real mu_beta;
  real<lower=0,upper=100> sigma;
  real<lower=0,upper=100> sigma_b;
  real<lower=0,upper=100> sigma_beta;
} 
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- b[county[i]] + x[i] * beta[1] + u[i] * beta[2];
}
model {
  mu_b ~ normal(0, 1);
  b ~ normal(mu_b, sigma_b);

  mu_beta ~ normal(0, 1);
  beta ~ normal(100 * mu_beta, sigma_beta);

  y ~ normal(y_hat, sigma);
}