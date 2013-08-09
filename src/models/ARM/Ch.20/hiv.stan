data {
  int<lower=0> J; 
  int<lower=0> N;
  int<lower=1,upper=J> person[N];
  vector[N] time;
  vector[N] y;
} 
parameters {
  matrix[2,J] a;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
}
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- a[1,person[i]] + a[2,person[i]] * time[i];
} 
model {
  mu_a ~ normal(0, 1);
  for (j in 1:2)
    a[j] ~ normal (2.5 * mu_a, sigma_a);

  y ~ normal(y_hat, sigma_y);
}
