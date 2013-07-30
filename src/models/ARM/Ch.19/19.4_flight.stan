data {
  int<lower=0> N; 
  int<lower=0> n_treatment; 
  int<lower=0> n_airport; 
  vector[N] y;
  vector[n_airport] airport;
  vector[n_treatment] treatment;
} 
parameters {
  vector[n_treatment] g;
  vector[n_airport] d;
  real<lower=0> sigma_y;
  real<lower=0> sigma_d;
  real<lower=0> sigma_g;
  real mu;
  real mu_d;
  real mu_g;
} 
transformed paramaters {
  vector[N] y_hat;
  vector[n_treatment] g_adj;
  vector[n_airport] d_adj;
  real mu_adj;
  real mu_g;
  real mu_d;
  mu_g <- mean(g)
  mu_d <- mean(d)
  g_adj <- g - mu_g;
  d_adj <- d - mu_d;
  mu_adj <- mu + mu_g + mu_d;
  for (i in 1:N)
    y_hat[i] <- mu + g[treatment[i]] + d[airport[i]];
}
model {
  sigma_y ~ uniform(0, 100);
  sigma_d ~ uniform(0, 100);
  sigma_g ~ uniform(0, 100);
  mu ~ normal(0, .0001);
  mu_g ~ normal(0, .0001);
  mu_d ~ normal(0, .0001);
  g ~ normal(mu_g, sigma_g);
  d ~ normal(mu_d, sigma_d);
  y ~ normal(y_hat, sigma_y);
}
