data {
  int<lower=0> N; 
  int<lower=1,upper=85> county[N];
  vector[N] u;
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[2] beta;
  vector[85] eta;
  real mu_b;
  real<lower=0,upper=100> sigma;
  real<lower=0,upper=100> sigma_b;
} 
transformed parameters {
  vector[85] b;
  vector[N] y_hat;

  b <- mu_b + sigma_b * eta;

  for (i in 1:N)
    y_hat[i] <- b[county[i]] + x[i] * beta[1] + u[i] * beta[2];
}
model {
  mu_b ~ normal(0, 1);
  eta ~ normal(0, 1);
  beta ~ normal(0, 100);

  y ~ normal(y_hat, sigma);
}
