data {
  int<lower=0> N; 
  int<lower=0> n_treatment; 
  int<lower=0> n_airport; 
  vector[N] y;
  int airport[N];
  int treatment[N];
} 
parameters {
  vector[n_treatment] g_raw;
  vector[n_airport] d_raw;
  real<lower=0> sigma_y;
  real<lower=0> sigma_d_raw;
  real<lower=0> sigma_g_raw;
  real mu;
  real<lower=0> xi_g;
  real xi_d;
  real mu_d_raw;
  real mu_g_raw;
} 
transformed parameters {
  vector[N] y_hat;
  vector[n_treatment] g;
  vector[n_airport] d;
  real<lower=0> sigma_d;
  real<lower=0> sigma_g;

  g <- xi_g * (g_raw - mean(g_raw));
  d <- xi_d * (d_raw - mean(d_raw));
  sigma_g <- xi_g * sigma_g_raw;
  sigma_d <- abs(xi_d) * sigma_d_raw;
  for (i in 1:N)
    y_hat[i] <- mu + g[treatment[i]] + d[airport[i]];
}
model {
  sigma_y ~ uniform(0, 100);
  sigma_d_raw ~ uniform(0, 100);
  sigma_g_raw ~ uniform(0, 100);
  xi_g ~ uniform(0, 100);
  xi_d ~ normal(0, .0001);
  mu ~ normal(0, .0001);
  mu_g_raw ~ normal(0, .0001);
  mu_d_raw ~ normal(0, .0001);
  g_raw ~ normal(mu_g_raw, sigma_g_raw);
  d_raw ~ normal(mu_d_raw, sigma_d_raw);
  y ~ normal(y_hat, sigma_y);
}
