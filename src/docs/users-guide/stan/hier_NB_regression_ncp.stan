functions {
  /*
  * Alternative to neg_binomial_2_log_rng() that 
  * avoids potential numerical problems during warmup
  */
  int neg_binomial_2_log_safe_rng(real eta, real phi) {
    real gamma_rate = gamma_rng(phi, phi / exp(eta));
    if (gamma_rate >= exp(20.79))
      return -9;     
    return poisson_rng(gamma_rate);
  }
}
data {
  int<lower=1> N;                     
  int<lower=0> complaints[N];              
  vector<lower=0>[N] traps;
  // 'exposure'
  vector[N] log_sq_foot;
  // building-level data
  int<lower=1> K;
  int<lower=1> J;
  int<lower=1, upper=J> building_idx[N];
  matrix[J,K] building_data;
}
parameters {
  real<lower=0> inv_phi;   // 1/phi (easier to think about prior for 1/phi instead of phi)
  real beta;               // coefficient on traps  
  vector[J] mu_raw;        // N(0,1) params for non-centered parameterization of building intercepts 
  real<lower=0> sigma_mu;  // sd of building-specific intercepts
  real alpha;              // intercept of model for mu
  vector[K] zeta;          // coefficients on building-level predictors in model for mu 
}
transformed parameters {
  real phi = inv(inv_phi); 
  // non-centered parameterization
  vector[J] mu = alpha + building_data * zeta + sigma_mu * mu_raw;
}
model {
  mu_raw ~ normal(0, 1);   // implies mu ~ normal(alpha + building_data * zeta, sigma_mu)
  sigma_mu ~ normal(0, 1);
  alpha ~ normal(log(4), 1);
  zeta ~ normal(0, 1);
  beta ~ normal(-0.25, 1);
  inv_phi ~ normal(0, 1);
  complaints ~ neg_binomial_2_log(mu[building_idx] + beta * traps + log_sq_foot, phi);
} 
generated quantities {
  int y_rep[N];
  for (n in 1:N) {
    real eta_n = mu[building_idx[n]] + beta * traps[n] + log_sq_foot[n];
    y_rep[n] = neg_binomial_2_log_safe_rng(eta_n, phi);
  }
}
