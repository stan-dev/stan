data {
  int<lower=0> N; 
  int<lower=0> n_state; 
  int y[N];
  vector[N] black;
  vector[N] female;
  int state[N];
} 
parameters {
  vector[n_state] a;
  vector[2] b;
  real<lower=0> sigma_a;
  real mu_a;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] <- b[1] * black[i] + b[2] * female[i] + a[state[i]];
} 
model {
  mu_a ~ normal(0, 100);
  sigma_a ~ uniform(0, 100);
  a ~ normal (mu_a, sigma_a);

  b ~ normal (0, 100);

  y ~ bernoulli_logit(y_hat);
}
