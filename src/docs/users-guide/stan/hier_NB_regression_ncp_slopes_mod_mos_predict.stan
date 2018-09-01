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
  // month 
  int<lower=1> M; 
  int<lower=1,upper=M> mo_idx[N];
  // for use in generated quantities
  int<lower=1> M_forward;
  vector[J] log_sq_foot_pred;
}
transformed data {
  // We'll make predictions for 0, 1, 2, ..., 20 traps (can go further too)
  int N_hypo_traps = 21;
  int hypo_traps[N_hypo_traps];
  for (i in 1:N_hypo_traps)
    hypo_traps[i] = i - 1;
}
parameters {
  real<lower=0> inv_phi;   // 1/phi (easier to think about prior for 1/phi instead of phi)
  vector[J] mu_raw;        // N(0,1) params for non-centered param of building-specific intercepts
  real<lower=0> sigma_mu;  // sd of buildings-specific intercepts
  real alpha;              // 'global' intercept
  vector[K] zeta;          // coefficients on building-level predictors in model for mu
  vector[J] kappa_raw;       // N(0,1) params for non-centered param of building-specific slopes
  real<lower=0> sigma_kappa; // sd of buildings-specific slopes
  real beta;                 // 'global' slope on traps variable
  vector[K] gamma;           // coefficients on building-level predictors in model for kappa
  vector[M] mo_raw;               // N(0,1) params for non-centered param of AR(1) process
  real<lower=0> sigma_mo;         // sd of month-specific parameters
  real<lower=0,upper=1> rho_raw;  // used to construct AR(1) coefficient
}
transformed parameters {
  real phi = inv(inv_phi);
  // non-centered parameterization of building-specific intercepts and slopes
  vector[J] mu = alpha + building_data * zeta + sigma_mu * mu_raw;
  vector[J] kappa = beta + building_data * gamma + sigma_kappa * kappa_raw;
  // AR(1) process priors
  real rho = 2 * rho_raw - 1;
  vector[M] mo = sigma_mo * mo_raw;
  mo[1] /= sqrt(1 - rho^2);
  for (m in 2:M) {
    mo[m] += rho * mo[m-1];
  }
}
model {
  inv_phi ~ normal(0, 1);
  kappa_raw ~ normal(0,1) ;
  sigma_kappa ~ normal(0, 1);
  beta ~ normal(-0.25, 1);
  gamma ~ normal(0, 1);
  mu_raw ~ normal(0,1) ;
  sigma_mu ~ normal(0, 1);
  alpha ~ normal(log(4), 1);
  zeta ~ normal(0, 1);
  mo_raw ~ normal(0,1);
  sigma_mo ~ normal(0, 1);
  rho_raw ~ beta(10, 5);
  { 
  // within a local block we can declare new temporary variables  
  vector[N] eta = mu[building_idx] + kappa[building_idx] .* traps + mo[mo_idx] + log_sq_foot;
  complaints ~ neg_binomial_2_log(eta, phi);
  /* 
  alternatively we can use a 'target +=' statement, which is equivalent to 
  above except that using '~' drops constants that don't affect inferences:
  target += neg_binomial_2_log_lpmf(complaints | eta, phi);
  */
  }
}
generated quantities {
  // we'll predict number of complaints and revenue lost for each building
  // at each hypothetical number of traps for M_forward months in the future
  int y_pred[J,N_hypo_traps];
  matrix[J,N_hypo_traps] rev_pred;
  for (j in 1:J) {
    for (i in 1:N_hypo_traps) {
      int y_pred_by_month[M_forward];
      vector[M_forward] mo_forward;
      mo_forward[1] = normal_rng(rho * mo[M], sigma_mo);
      for (m in 2:M_forward) 
        mo_forward[m] = normal_rng(rho * mo_forward[m-1], sigma_mo); 
      for (m in 1:M_forward) {
        real eta = mu[j] + kappa[j] * hypo_traps[i] + mo_forward[m] + log_sq_foot_pred[j];
        y_pred_by_month[m] = neg_binomial_2_log_safe_rng(eta, phi);
      }
      y_pred[j,i] = sum(y_pred_by_month);
      // were were told every 10 complaints has additional exterminator cost of $100, 
      // so $10 lose per complaint.
      rev_pred[j,i] = -10 * y_pred[j,i];
    }
  }
}
