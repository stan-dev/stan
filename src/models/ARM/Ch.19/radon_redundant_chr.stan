data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=1,upper=J> county[N];
  vector[N] y;
} 
parameters {
  vector[J] et;
  real mu_eta;
  real<lower=0,upper=100> sigma_eta;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  vector[J] eta;
  vector[J] eta_adj;
  real mean_eta;
  real mu_adj;
  vector[N] y_hat;

  eta <- 100 * mu_eta + sigma_eta * et;

  mean_eta <- mean(eta);
  mu_adj <- mean_eta;
  eta_adj <- eta - mean_eta;
  for (i in 1:N)
    y_hat[i] <- eta[county[i]];
}
model {
  mu_eta ~ normal(0, 1);
  sigma_eta ~ uniform(0, 100);
  sigma_y ~ uniform(0, 100);
  et ~ normal (0, 1);

  y ~ normal(y_hat,sigma_y);
}
