data {
  int<lower=0> N; 
  int<lower=0> J;
  int<lower=0> K;
  vector[N] y;
  vector[N] x;
  vector[N] k;
  vector[N] j;
} 
parameters {
  vector[J] a_raw;
  real mu_a_raw;
  real<lower=0> sigma_a_raw;
  vector[k] b_raw;
  real<lower=0> sigma_b_raw;
  vector[K] g_raw;
  real mu_g_raw;
  real<lower=0> sigma_g_raw;
  real b_0_raw;
  real<lower=0> d_raw;
} 
transformed paramaters {
  real shift;
  real scale;
  vector[N] p;
  vector[N] p_bound;
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
  g <- g_raw[k] * scale;
  d <- d_raw * scale;
  for (i in 1:n){
    p[i] <- inv_logit(g[k[i]]*(a[j[i]] - b[k[i]]));
    p_bound[i] <- max(0, min(1, p[i]));
  }
}
model {
  mu_a_raw ~ normal(0, .0001);
  sigma_a_raw ~ uniform(0, 100);
  a_raw ~ normal(mu_a_raw, sigma_a_raw);

  sigma_b_raw ~ uniform(0,100);
  b_raw ~ normal(b_hat_raw, sigma_b_raw);

  mu_g_raw ~ normal(0, .0001);
  sigma_g_raw ~ uniform(0, 100);
  g_raw ~ normal(mu_g_raw, sigma_g_raw);
  b_0_raw ~ normal(0, .0001);
  d_raw ~ normal(0, .0001);

  y ~ binomial(p_bound,1);
}
