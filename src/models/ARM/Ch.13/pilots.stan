data {
  int<lower=0> N; 
  int<lower=0> n_scenarios; 
  int<lower=0> n_groups; 
  vector[N] y;
  int group_id[N];
  int scenario_id[N];
} 
parameters {
  vector[n_groups] eta_a;
  vector[n_scenarios] eta_b;
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real mu_a;
  real<lower=0> sigma_b;
  real mu_b;
}
transformed parameters {
  vector[N] y_hat;
  vector[n_groups] a;
  vector[n_scenarios] b;

  a <- mu_a + eta_a * sigma_a;
  b <- mu_b + eta_b * sigma_b;

  for (i in 1:N)
    y_hat[i] <- a[group_id[i]] + b[scenario_id[i]];
} 
model {
  mu_a ~ normal(0, 100);
  eta_a ~ normal(0, 1);

  mu_b ~ normal(0, 100);
  eta_b ~ normal(0, 1);

  y ~ normal(y_hat, sigma_y);
}
