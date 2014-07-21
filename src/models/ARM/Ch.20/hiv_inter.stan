data {
  int<lower=0> J; 
  int<lower=0> N;
  int<lower=1,upper=J> person[N];
  vector[N] time;
  vector[N] treatment;
  vector[N] y;
} 
parameters {
  vector[J] a1;
  vector[J] a2;
  real beta;
  real mu_a1;
  real mu_a2;
  real<lower=0> sigma_a1;
  real<lower=0> sigma_a2;
  real<lower=0> sigma_y;
}
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- beta * time[i] * treatment[i] + a1[person[i]] 
                + a2[person[i]] * time[i];
} 
model {
  mu_a1 ~ normal(0, 1);
  a1 ~ normal (10 * mu_a1, sigma_a1);
  mu_a2 ~ normal(0, 1);
  a2 ~ normal (0.1 * mu_a2, sigma_a2);

  beta ~ normal (0, 1);

  y ~ normal(y_hat, sigma_y);
}
