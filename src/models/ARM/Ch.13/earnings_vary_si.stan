data {
  int<lower=0> N; 
  vector[N] earn;
  int eth[N];
  vector[N] height;
} 
transformed data {
  vector[N] log_earn;

  log_earn <- log(earn);
}
parameters {
  vector[4] a1;
  vector[4] a2;
  real<lower=0> sigma_a1;
  real<lower=0> sigma_a2;
  real<lower=0> sigma_y;
  real mu_a1;
  real mu_a2;
}
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- a1[eth[i]] + a2[eth[i]] * height[i];
} 
model {
  mu_a1 ~ normal(0, 1);
  mu_a2 ~ normal(0, 1);
  a1 ~ normal(10 * mu_a1, sigma_a1);
  a2 ~ normal(0.01 * mu_a2, sigma_a2);
  sigma_a1 ~ cauchy(0, 5);
  sigma_a2 ~ cauchy(0, 5);
  sigma_y ~ cauchy(0, 5);
  log_earn ~ normal(y_hat, sigma_y);
}