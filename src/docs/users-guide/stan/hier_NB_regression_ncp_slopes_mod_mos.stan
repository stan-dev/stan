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
  complaints ~ neg_binomial_2_log(mu[building_idx] + kappa[building_idx] .* traps 
    + mo[mo_idx] + log_sq_foot, phi);
}
generated quantities {
  int y_rep[N];
  for (n in 1:N) {
    real eta_n = mu[building_idx[n]] + kappa[building_idx[n]] * traps[n] + mo[mo_idx[n]]
      + log_sq_foot[n];
    y_rep[n] = neg_binomial_2_log_safe_rng(eta_n, phi);
  }
}
