data {
  int<lower=0> J; 
  int<lower=0> N;
  int<lower=1,upper=J> person[N];
  vector[N] time;
  vector[N] y;
} 
parameters {
  matrix[2,J] eta;
  real mu_a;
  real<lower=0> sigma_a;
  real<lower=0> sigma_y;
}
transformed parameters {
  matrix[2,J] a;
  vector[N] y_hat;

  a <- 100 * mu_a + sigma_a * eta;

  for (i in 1:N)
    y_hat[i] <- a[1,person[i]] + a[2,person[i]] * time[i];
} 
model {
  mu_a ~ normal(0, 1);
  for (j in 1:2)
    eta[j] ~ normal (0,1);

  y ~ normal(y_hat, sigma_y);
}
