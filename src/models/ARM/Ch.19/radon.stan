data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=1,upper=J> county[N];
  vector[N] y;
} 
parameters {
  vector[J] eta;
  real mu;
  real<lower=0,upper=100> sigma_eta;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- 0.1 * mu + eta[county[i]];
}
model {
  sigma_eta ~ uniform(0, 100);
  sigma_y ~ uniform(0, 100);
  mu ~ normal(0, 1);
  eta ~ normal (0, sigma_eta);
  y ~ normal(y_hat,sigma_y);
}
