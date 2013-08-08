data {
  int<lower=0> N; 
  int<lower=0> n_age; 
  int<lower=0> n_eth; 
  int<lower=1,upper=n_age> age[N];
  int<lower=1,upper=n_eth> eth[N];
  vector[N] x_centered;
  vector[N] y;
} 
parameters {
  matrix[2,n_eth] eta_a;
  matrix[2,n_age] eta_b;
  matrix[n_eth,n_age] eta_c;
  matrix[n_eth,n_age] eta_d;
  real mu_a;
  real mu_b;
  real mu_c;
  real mu_d;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_b;
  real<lower=0,upper=100> sigma_c;
  real<lower=0,upper=100> sigma_d;
  real<lower=0,upper=100> sigma_y;
}
transformed parameters {
  matrix[2,n_eth] a;
  matrix[2,n_age] b;
  matrix[n_eth,n_age] c;
  matrix[n_eth,n_age] d;
  vector[N] y_hat;

  a <- 100 * mu_a + sigma_a * eta_a;
  b <- 100 * mu_b + sigma_b * eta_b;
  c <- 100 * mu_c + sigma_c * eta_c;
  d <- 100 * mu_d + sigma_d * eta_d;

  for (i in 1:N)
    y_hat[i] <- a[1,eth[i]] + a[2,eth[i]] * x_centered[i] + b[1,age[i]] 
                + b[2,age[i]] * x_centered[i] + c[eth[i],age[i]] 
                + d[eth[i],age[i]] * x_centered[i];
} 
model {
  mu_a ~ normal(0, 1);
  for (j in 1:2)
    eta_a[j] ~ normal(0, 1);

  mu_b ~ normal(0, 1);
  for (j in 1:2)
    eta_b[j] ~ normal(0, 1);

  mu_c ~ normal(0, 1);
  for (j in 1:n_eth)
    eta_c[j] ~ normal(0, 1);

  mu_d ~ normal(0, 1);
  for (j in 1:n_eth)
    eta_d[j] ~ normal(0, 1);

  y ~ normal(y_hat, sigma_y);
}
