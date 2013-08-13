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
  vector[N] y_hat;

  eta <- 0.1 * mu_eta + sigma_eta * et;

  for (i in 1:N)
    y_hat[i] <- eta[county[i]];
}
model {
  et ~ normal (0, 1);
  mu_eta ~ normal(0, 1);
  y ~ normal(y_hat,sigma_y);
}
