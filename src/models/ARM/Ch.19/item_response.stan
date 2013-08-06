data {
  int<lower=0> N; 
  int<lower=0> J;
  int<lower=0> K;
  int y[N];
  vector[N] x;
  int k[N];
  int j[N];
} 
parameters {
  vector[J] a_raw;
  real mu_a_raw;
  real<lower=0> sigma_a_raw;
  vector[K] b_raw;
  real<lower=0> sigma_b_raw;
  vector[K] g_raw;
  real mu_g_raw;
  real<lower=0> sigma_g_raw;
  real b_0_raw;
  real<lower=0> d_raw;
} 
transformed parameters {
  real shift;
  real scale;
  vector[N] p;
  vector[J] a;
  vector[K] b_hat_raw;
  vector[K] b;
  vector[K] g;
  real d;

  shift <- mean(a);
  scale <- sd(a);
  a <- (a_raw - shift) / scale;
  b_hat_raw <- b_0_raw + d_raw * x;
  b <- (b_raw - shift) / scale;
  g <- g_raw * scale;
  d <- d_raw * scale;
  for (i in 1:N)
    p[i] <- (g[k[i]]*(a[j[i]] - b[k[i]]));
}
model {
  mu_a_raw ~ normal(0, 100);
  sigma_a_raw ~ uniform(0, 100);
  a_raw ~ normal(mu_a_raw, sigma_a_raw);

  sigma_b_raw ~ uniform(0,100);
  b_raw ~ normal(b_hat_raw, sigma_b_raw);

  mu_g_raw ~ normal(0, 100);
  sigma_g_raw ~ uniform(0, 100);
  g_raw ~ normal(mu_g_raw, sigma_g_raw);
  b_0_raw ~ normal(0, 100);
  d_raw ~ normal(0, 100);

  y ~ bernoulli_logit(p);
}
