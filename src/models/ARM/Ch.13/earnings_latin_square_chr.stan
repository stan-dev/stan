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
  vector[n_eth] eta_a1;
  vector[n_eth] eta_a2;
  vector[n_age] eta_b1;
  vector[n_age] eta_b2;
  matrix[n_eth,n_age] eta_c;
  matrix[n_eth,n_age] eta_d;
  real mu_a1;
  real mu_a2;
  real mu_b1;
  real mu_b2;
  real mu_c;
  real mu_d;
  real<lower=0,upper=100> sigma_a1;
  real<lower=0,upper=100> sigma_a2;
  real<lower=0,upper=100> sigma_b1;
  real<lower=0,upper=100> sigma_b2;
  real<lower=0,upper=100> sigma_c;
  real<lower=0,upper=100> sigma_d;
  real<lower=0,upper=100> sigma_y;
}
transformed parameters {
  vector[n_eth] a1;
  vector[n_eth] a2;
  vector[n_age] b1;
  vector[n_age] b2;
  matrix[n_eth,n_age] c;
  matrix[n_eth,n_age] d;
  vector[N] y_hat;

  a1 <- 5 * mu_a1 + sigma_a1 * eta_a1;
  a2 <- mu_a2 + sigma_a2 * eta_a2;
  b1 <- 5 * mu_b1 + sigma_b1 * eta_b1;
  b2 <- 0.1 * mu_b2 + sigma_b2 * eta_b2;
  c <- 0.1 * mu_c + sigma_c * eta_c;
  d <- 0.01 * mu_d + sigma_d * eta_d;

  for (i in 1:N)
    y_hat[i] <- a1[eth[i]] + a2[eth[i]] * x_centered[i] + b1[age[i]] 
                + b2[age[i]] * x_centered[i] + c[eth[i],age[i]] 
                + d[eth[i],age[i]] * x_centered[i];
} 
model {
  mu_a1 ~ normal(0, 1);
  mu_a2 ~ normal(0, 1);
  eta_a1 ~ normal(0, 1);
  eta_a2 ~ normal(0, 1);

  mu_b1 ~ normal(0, 1);
  mu_b2 ~ normal(0, 1);
  eta_b1 ~ normal(0, 1);
  eta_b2 ~ normal(0, 1);

  mu_c ~ normal(0, 1);
  for (j in 1:n_eth)
    eta_c[j] ~ normal(0, 1);

  mu_d ~ normal(0, 1);
  for (j in 1:n_eth)
    eta_d[j] ~ normal(0, 1);

  y ~ normal(y_hat, sigma_y);
}
