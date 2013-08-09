data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=1,upper=J> county[N];
  vector[J] u;
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[J] a;
  vector[2] b;
  real mu_b;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_b;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- a[county[i]] + x[i] * b[1] + u[i] * b[2];
}
model {
  sigma_a ~ uniform(0, 100);
  a ~ normal (a_hat, sigma_a);

  mu_b ~ normal(0, 1);
  sigma_b ~ uniform(0, 100);
  b ~ normal (100 * mu_b, sigma_b);

  sigma_y ~ uniform(0, 100);
  y ~ normal(y_hat, sigma_y);
}
