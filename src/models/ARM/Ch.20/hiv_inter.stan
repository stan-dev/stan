data {
  int<lower=0> N;
  int<lower=0> J; 
  vector[N] y;
  vector[N] time;
  vector[N] treatment;
  int person[N];
} 
parameters {
  vector[J] a;
  vector[J] b;
  real beta;
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real mu_a;
  real<lower=0> sigma_b;
  real mu_b;
  real<lower=0> sigma_beta;
  real mu_beta;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] <- beta * time[i] * treatment[i] + a[person[i]] 
                + b[person[i]] * time[i];
} 
model {
  mu_a ~ normal(0, .0001);
  sigma_a ~ uniform(0, 100);
  a ~ normal (mu_a, sigma_a);

  mu_b ~ normal(0, .0001);
  sigma_b ~ uniform(0, 100);
  b ~ normal (mu_b, sigma_b);

  mu_beta ~ normal(0, .0001);
  sigma_beta ~ uniform(0, 100);
  beta ~ normal (mu_beta, sigma_beta);

  sigma_y ~ uniform(0, 100);
  y ~ normal(y_hat, sigma_y);
}
