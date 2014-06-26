data {
  int<lower=0> K;
  int<lower=0> J;
  int<lower=0> N; 
  int<lower=0,upper=J> j[N];
  int<lower=1,upper=K> k[N];
  vector[N] x;
  int<lower=0,upper=1> y[N];
} 
parameters {
  vector[J] a_raw;
  vector[K] b_raw;
  vector[K] g_raw;
  real b_0_raw;
  real mu_a_raw;
  real mu_g_raw;
  real<lower=0> d_raw;
  real<lower=0,upper=100> sigma_a_raw;
  real<lower=0,upper=100> sigma_b_raw;
  real<lower=0,upper=100> sigma_g_raw;
} 
transformed parameters {
  vector[J] a;
  vector[K] b;
  vector[K] b_hat_raw;
  real d;
  vector[K] g;
  vector[N] p;
  real scale;
  real shift;

  shift <- mean(a_raw);
  scale <- sd(a_raw);
  a <- (a_raw - shift) / scale;
  b_hat_raw <- 100 * b_0_raw + d_raw * x;
  b <- (b_raw - shift) / scale;
  g <- g_raw * scale;
  d <- d_raw * scale;
  for (i in 1:N)
    p[i] <- (g[k[i]]*(a[j[i]] - b[k[i]]));
}
model {
  mu_a_raw ~ normal(0, 1);
  sigma_a_raw ~ uniform(0, 100);
  a_raw ~ normal(100 * mu_a_raw, sigma_a_raw);

  sigma_b_raw ~ uniform(0,100);
  b_raw ~ normal(b_hat_raw, sigma_b_raw);

  mu_g_raw ~ normal(0, 1);
  sigma_g_raw ~ uniform(0, 100);
  g_raw ~ normal(100 * mu_g_raw, sigma_g_raw);
  b_0_raw ~ normal(0, 1);
  d_raw ~ normal(0, 100);

  y ~ bernoulli_logit(p);
}
