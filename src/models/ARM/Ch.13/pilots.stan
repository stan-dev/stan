data {
  int<lower=0> N; 
  int<lower=0> n_groups; 
  int<lower=0> n_scenarios; 
  int<lower=1,upper=n_groups> group_id[N];
  int<lower=1,upper=n_scenarios> scenario_id[N];
  vector[N] y;
} 
parameters {
  vector[n_groups] a;
  vector[n_scenarios] b;
  real mu_a;
  real mu_b;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_b;
  real<lower=0,upper=100> sigma_y;
}
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- a[group_id[i]] + b[scenario_id[i]];
} 
model {
  mu_a ~ normal(0, 1);
  mu_b ~ normal(0, 1);
  a ~ normal(0, sigma_a);
  b ~ normal(0, sigma_b);
  y ~ normal(y_hat, sigma_y);
}