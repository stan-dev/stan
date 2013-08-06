data {
  int<lower=0> N; 
  vector[N] earn;
  vector[N] height;
  int eth[N];
} 
transformed data {
  vector[N] log_earn;
  log_earn <- log(earn);
}
parameters {
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real mu_a;
  matrix[4,2] eta;
}
transformed parameters {
  vector[N] y_hat;
  matrix[4,2] a;

  a <- mu_a + sigma_a * eta;

  for (i in 1:N)
    y_hat[i] <- a[eth[i],1] + a[eth[i],2] * height[i];
} 
model {
  mu_a ~ normal(0, 100);
  for (j in 1:4)
    eta[j] ~ normal(0, 1);


  log_earn ~ normal(y_hat, sigma_y);
}
