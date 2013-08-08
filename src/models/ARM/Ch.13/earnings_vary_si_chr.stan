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
  matrix[2,4] eta;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
}
transformed parameters {
  matrix[2,4] a;
  vector[N] y_hat;

  a <- 100 * mu_a + sigma_a * eta;

  for (i in 1:N)
    y_hat[i] <- a[1,eth[i]] + a[2,eth[i]] * height[i];
} 
model {
  mu_a ~ normal(0, 1);
  for (j in 1:2)
    eta[j] ~ normal(0, 1);


  log_earn ~ normal(y_hat, sigma_y);
}
