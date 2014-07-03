data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=1,upper=J> county[N];
  vector[N] u;
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[2] b;
  vector[J] eta;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  vector[J] a;
  vector[N] y_hat;

  a <- mu_a + sigma_a * eta;

  for (i in 1:N)
    y_hat[i] <- a[county[i]] + u[i] * b[1] + x[i] * b[2];
}
model {
  mu_a ~ normal(0, 1);
  eta ~ normal(0, 1);
  b ~ normal(0, 1);
  y ~ normal(y_hat, sigma_y);
}
