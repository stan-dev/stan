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
  vector[N] y_hat;
  vector[J] eta;

  eta <- mu_eta + sigma_eta * et;

  for (i in 1:N)
    y_hat[i] <- mu + eta[county[i]];
}
model {
  sigma_y ~ uniform(0, 100);
  mu ~ normal(0, .0001);
  mu_eta ~ normal(0, 100);
  sigma_eta ~ uniform(0, 100);
  et ~ normal (0, 1);

  y ~ normal(y_hat,sigma_y);
}
