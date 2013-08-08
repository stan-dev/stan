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
  matrix[2,4] a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
  real mu_a;
}
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- a[1,eth[i]] + a[2,eth[i]] * height[i];
} 
model {
  mu_a ~ normal(0, 1);
  for (i in 1:2)
    a[i] ~ normal (100 * mu_a, sigma_a);

  log_earn ~ normal(y_hat, sigma_y);
}