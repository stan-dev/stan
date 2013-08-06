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
  real mu_eta;
  vector[J] et;
} 
transformed parameters {
  vector[J] eta;
  vector[N] y_hat;
  real mu_adj;
  vector[J] eta_adj;
  real mean_eta;

  eta <- mu_eta + sigma_eta * et;

  mean_eta <- mean(eta);
  mu_adj <- mu + mean_eta;
  eta_adj <- eta - mean_eta;
  for (i in 1:N)
    y_hat[i] <- mu + eta[county[i]];
}
model {
  mu_eta ~ normal(0, 100);
  sigma_eta ~ uniform(0, 100);
  sigma_y ~ uniform(0, 100);
  mu ~ normal(0, 100);
  et ~ normal (0, 1);

  y ~ normal(y_hat,sigma_y);
}
