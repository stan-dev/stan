data {
  int<lower=0> N; 
  int<lower=1,upper=85> county[N];
  vector[N] x;
  vector[N] y;
} 
parameters {
  matrix[2,85] a;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
}
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- a[1,county[i]] + a[2,county[i]] * x[i];
} 
model {
  mu_a ~ normal(0, 1);
  for (j in 1:2)
    a[j] ~ normal(100 * mu_a, sigma_a);

  y ~ normal(y_hat, sigma_y);
}
