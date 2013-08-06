data {
  int<lower=0> N; 
  vector[N] y;
  vector[N] x;
  vector[N] u;
  int county[N];
} 
parameters {
  vector[2] beta;
  real<lower=0> sigma;
  real mu_b;
  real<lower=0> sigma_b;
  vector[85] eta;
} 
transformed parameters {
  vector[N] y_hat;
  vector[85] b;

  b <- mu_b + sigma_b * eta;
  for (i in 1:N)
    y_hat[i] <- b[county[i]] + x[i] * beta[1] + u[i] * beta[2];
}
model {
  mu_b ~ normal(0, 100);
  eta ~ normal(0, 1);
  beta ~ normal(0, 100);

  y ~ normal(y_hat, sigma);
}
