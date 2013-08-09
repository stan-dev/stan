data {
  int<lower=0> J; 
  int<lower=0> N;
  int<lower=1,upper=J> person[N];
  vector[N] time;
  vector[N] treatment;
  vector[N] y;
} 
parameters {
  matrix[2,J] a;
  real beta;
  real mu_a;
  real<lower=0> sigma_a;
  real<lower=0> sigma_y;
}
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- 100 * beta * time[i] * treatment[i] + a[1,person[i]] 
                + a[2,person[i]] * time[i];
} 
model {
  mu_a ~ normal(0, 1);
  for (j in 1:2)
    a[j] ~ normal (100 * mu_a, sigma_a);

  beta ~ normal (0, 1);

  y ~ normal(y_hat, sigma_y);
}
