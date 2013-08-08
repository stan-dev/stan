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
  vector[2] beta;
  matrix[2,85] eta;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  matrix[2,85] a;
  vector[N] y_hat;

  a <- 100 * mu_a + sigma_a * eta;

  for (i in 1:N)
    y_hat[i] <- a[1,county[i]] + x[i] * a[2,county[i]] + beta[1] * u[i] 
                + beta[2] * inter[i];
}
model {
  beta ~ normal(0, 100);

  mu_a ~ normal(0, 1);
  for (j in 1:2)
    eta[j] ~ normal(0, 1);

  y ~ normal(y_hat, sigma_y);
}
