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
  vector[4] a;
  vector[4] b;
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real mu_a;
  real<lower=0> sigma_b;
  real mu_b;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] <- a[ethn[i]] + b[ethn[i]] * height[i];
} 
model {
  mu_a ~ normal(0, 100);
  sigma_a ~ uniform(0, 100);
  a ~ normal (mu_a, sigma_a);

  mu_b ~ normal(0, 100);
  sigma_b ~ uniform(0, 100);
  b ~ normal (mu_b, sigma_b);

  sigma_y ~ uniform(0, 100);
  log_earn ~ normal(y_hat, sigma_y);
}