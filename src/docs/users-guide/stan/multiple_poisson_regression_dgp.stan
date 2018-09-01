data {
  int<lower=1> N;
}
model {
} 
generated quantities {
  vector[N] log_sq_foot;
  int live_in_super[N];
  int traps[N];
  int complaints[N];
  real alpha = normal_rng(log(4), 0.1);
  real beta = normal_rng(-0.25, 0.1);
  real beta_super = normal_rng(-0.5, 0.1);
  for (n in 1:N) {
    log_sq_foot[n] = normal_rng(1.5, 0.1);
    live_in_super[n] = bernoulli_rng(0.5);
    traps[n] = poisson_rng(8);
    complaints[n] = poisson_log_rng(alpha + log_sq_foot[n] 
      + beta * traps[n] + beta_super * live_in_super[n]);
  }
}
