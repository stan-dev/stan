data {
  int<lower=0> N; 
  int<lower=0> n_scenarios; 
  int<lower=0> n_groups; 
  vector[N] y;
  int group_id[N];
  int scenario_id[N];
} 
parameters {
  vector[n_groups] a;
  vector[n_scenarios] b;
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real mu_a;
  real<lower=0> sigma_b;
  real mu_b;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] <- a[group_id[i]] + b[scenario_id[i]];
} 
model {
  mu_a ~ normal(0, 100);
  a ~ normal (mu_a, sigma_a);

  mu_b ~ normal(0, 100);
  b ~ normal (mu_b, sigma_b);

  y ~ normal(y_hat, sigma_y);
}