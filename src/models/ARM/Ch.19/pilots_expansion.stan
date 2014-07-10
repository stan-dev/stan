data {
  int<lower=0> N; 
  int<lower=0> n_airport; 
  int<lower=0> n_treatment; 
  int<lower=1,upper=n_airport> airport[N];
  int<lower=1,upper=n_treatment> treatment[N];
  vector[N] y;
} 
parameters {
  vector[n_airport] d_raw;
  vector[n_treatment] g_raw;
  real mu;
  real mu_d_raw;
  real mu_g_raw;
  real<lower=0,upper=100> sigma_d_raw;
  real<lower=0,upper=100> sigma_g_raw;
  real<lower=0,upper=100> sigma_y;
  real xi_d;
  real<lower=0,upper=100> xi_g;
} 
transformed parameters {
  vector[n_airport] d;
  vector[n_treatment] g;
  real<lower=0> sigma_d;
  real<lower=0> sigma_g;
  vector[N] y_hat;

  g <- xi_g * (g_raw - mean(g_raw));
  d <- xi_d * (d_raw - mean(d_raw));
  sigma_g <- xi_g * sigma_g_raw;
  sigma_d <- fabs(xi_d) * sigma_d_raw;
  for (i in 1:N)
    y_hat[i] <- mu + g[treatment[i]] + d[airport[i]];
}
model {
  sigma_y ~ uniform(0, 100);
  sigma_d_raw ~ uniform(0, 100);
  sigma_g_raw ~ uniform(0, 100);
  xi_g ~ uniform(0, 100);
  xi_d ~ normal(0, 100);
  mu ~ normal(0, 100);
  mu_g_raw ~ normal(0, 1);
  mu_d_raw ~ normal(0, 1);
  g_raw ~ normal(100 * mu_g_raw, sigma_g_raw);
  d_raw ~ normal(100 * mu_d_raw, sigma_d_raw);
  y ~ normal(y_hat, sigma_y);
}
