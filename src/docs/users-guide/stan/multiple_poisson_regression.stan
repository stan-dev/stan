functions {
  /*
  * Alternative to poisson_log_rng() that 
  * avoids potential numerical problems during warmup
  */
  int poisson_log_safe_rng(real eta) {
    real pois_rate = exp(eta);
    if (pois_rate >= exp(20.79))
      return -9;
    return poisson_rng(pois_rate);
  }
}
data {
  int<lower=1> N;
  int<lower=0> complaints[N];
  vector<lower=0>[N] traps;
  vector<lower=0,upper=1>[N] live_in_super;
  vector[N] log_sq_foot;  // exposure term
}
parameters {
  real alpha;
  real beta;
  real beta_super;
}
model {
  beta ~ normal(-0.25, 1);
  beta_super ~ normal(-0.5, 1);
  alpha ~ normal(log(4), 1);  
  complaints ~ poisson_log(alpha + beta * traps + beta_super * live_in_super + log_sq_foot);
} 
generated quantities {
  int y_rep[N]; 
  for (n in 1:N) 
    y_rep[n] = poisson_log_safe_rng(alpha + beta * traps[n] + beta_super * live_in_super[n]
                                   + log_sq_foot[n]);
}
