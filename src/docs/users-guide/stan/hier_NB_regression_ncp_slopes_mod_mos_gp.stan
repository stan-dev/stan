functions {
  int neg_binomial_2_log_safe_rng(real eta, real phi) {
    real phi_div_exp_eta;
    real gamma_rate;
    phi_div_exp_eta = phi/exp(eta);
    gamma_rate = gamma_rng(phi, phi_div_exp_eta);
    if (gamma_rate >= exp(20.79))
      return -9;
    return poisson_rng(gamma_rate);
  }
}
data {
  int<lower=1> N;
  int<lower=1> M;
  int<lower=1> K;
  int complaints[N];
  vector[N] traps;
  int<lower=1> J;
  int<lower=1, upper=J> building_idx[N];
  matrix[J,K] building_data;
  vector[N] log_sq_foot;
  int<lower=1> mo_idx[N];
}
transformed data {
  real mo_gp_vec[M];
  for (m in 1:M)
    mo_gp_vec[m] = m;
}
parameters {
  real alpha;
  real<lower=0> sigma_mu;
  real<lower=0> sigma_kappa;
  vector[J] mu_raw;
  vector[J] kappa_raw;
  real beta;
  real<lower=0> inv_phi;
  vector[K] zeta;
  vector[K] gamma;
  // GP prior parameters
  vector[M] gp_raw;
  real<lower=0> gp_len;
  real<lower=0> sigma_gp;
  real<lower=0> sigma_noise;
  vector[M] mo_noise_raw;
}
transformed parameters {
  vector[J] mu = alpha + building_data * zeta + sigma_mu * mu_raw;
  vector[J] kappa = beta + building_data * gamma + sigma_kappa * kappa_raw;
  vector[M] mo_noise = sigma_noise * mo_noise_raw;
  real phi = inv(inv_phi);
  vector[M] gp_exp_quad;
  vector[M] gp;
  {
    matrix[M, M] C = cov_exp_quad(mo_gp_vec, sigma_gp, gp_len);
    real var_noise = square(sigma_noise);
    matrix[M, M] L_C;
    for (m in 1:M)
      C[m,m] += 1e-12;
    L_C = cholesky_decompose(C);
    gp_exp_quad = L_C * gp_raw;
  }
  
  // gp is sum of monthly noise and the smoothly varying process
  gp = mo_noise + gp_exp_quad;
}
model {
  beta ~ normal(-0.25, 1);
  mu_raw ~ normal(0,1);
  kappa_raw ~ normal(0,1);
  sigma_mu ~ normal(0, 1);
  sigma_kappa ~ normal(0, 1);
  alpha ~ normal(log(4), 1);
  zeta ~ normal(0, 1);
  gamma ~ normal(0, 1);
  inv_phi ~ normal(0, 1);
  
  // GP priors
  gp_raw ~ normal(0, 1);
  gp_len ~ gamma(10, 2);
  sigma_gp ~ normal(0, 1);
  
  sigma_noise ~ normal(0, 1);
  mo_noise_raw ~ normal(0, 1);
  
  complaints ~ neg_binomial_2_log(mu[building_idx] + kappa[building_idx] .* traps 
                                 + gp[mo_idx] + log_sq_foot, phi);
} 
generated quantities {
  int y_rep[N];

  for (n in 1:N) 
    y_rep[n] = neg_binomial_2_log_safe_rng(mu[building_idx[n]] + kappa[building_idx[n]] * traps[n]
                                          + gp[mo_idx[n]] + log_sq_foot[n],
                                          phi);
}
