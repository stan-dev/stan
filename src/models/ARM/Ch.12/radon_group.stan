data {
  int<lower=0> N; 
  vector[N] y;
  vector[N] x;
  vector[N] u;
  int county[N];
} 
parameters {
  vector[85] b;
  vector[2] beta;
  real<lower=0> sigma;
  real mu_b;
  real<lower=0> sigma_b;
  real mu_beta;
  real<lower=0> sigma_beta;
} 
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] <- b[county[i]] + x[i] * beta[1] + u[i] * beta[2];
}
model {
  mu_b ~ normal(0, .0001);
  mu_beta ~ normal(0, .0001);
  sigma_b ~ uniform(0, 100);
  sigma_beta ~ uniform(0, 100);
  sigma ~ uniform(0, 100);
  b ~ normal(mu_b, sigma_b);
  beta ~ normal(mu_beta, sigma_beta);
  y ~ normal(y_hat, sigma);
}
