data {
  int<lower=0> N; 
  int<lower=0> J; 
  vector[N] y;
  int county[N];
} 
parameters {
  real<lower=0> sigma_eta;
  real<lower=0> sigma_y;
  real mu;
  vector[J] eta;
} 
transformed parameters {
  vector[N] y_hat;
  real mu_adj;
  vector[J] eta_adj;
  real mean_eta;
  mean_eta <- mean(eta);
  mu_adj <- mu + mean_eta;
  eta_adj <- eta - mean_eta;
  for (i in 1:N)
    y_hat[i] <- mu + eta[county[i]];
}
model {
  sigma_eta ~ uniform(0, 100);
  sigma_y ~ uniform(0, 100);
  mu ~ normal(0, 100);
  eta ~ normal (0, sigma_eta);
  y ~ normal(y_hat,sigma_y);
}
