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
  int<lower=1> K;
  int complaints[N];
  vector[N] traps;
  int<lower=1> J;
  int<lower=1, upper=J> building_idx[N];
  matrix[J,K] building_data;
  vector[N] log_sq_foot;
  int mo_idx[N];
  int M;
}
parameters {
  real alpha;
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;
  vector[J] std_alphas;
  vector[J] std_betas;
  real beta;
  real<lower=0> inv_prec;
  vector[K] zeta;
  vector[K] gamma;
}
transformed parameters {
  vector[J] alphas = alpha + building_data * zeta + sigma_alpha * std_alphas;
  vector[J] betas = beta + building_data * gamma + sigma_beta * std_betas;
  real prec = inv(inv_prec);
}
model {
  beta ~ normal(0, 1);
  std_alphas ~ normal(0,1) ;
  std_betas ~ normal(0,1) ;
  sigma_alpha ~ normal(0, 1);
  sigma_beta ~ normal(0, 1);
  alpha ~ normal(log(4), 1);
  zeta ~ normal(0, 1);
  gamma ~ normal(0, 1);
  inv_prec ~ normal(0, 1);
  complaints ~ neg_binomial_2_log(alphas[building_idx] + betas[building_idx] .* traps 
    + log_sq_foot, prec);
} 
generated quantities {
  int y_rep[N];
  vector[N] std_resid;
  vector[M] mo_resid = rep_vector(0, M);
  for (n in 1:N) {
    real eta = alphas[building_idx[n]] + betas[building_idx[n]] * traps[n]
      + log_sq_foot[n];
    y_rep[n] = neg_binomial_2_log_safe_rng(eta, prec);
    std_resid[n] = (y_rep[n] - eta) / sqrt(exp(eta) + exp(eta)^2 * inv_prec);
    mo_resid[mo_idx[n]] = mo_resid[mo_idx[n]] + std_resid[n];
  }
  for (m in 1:M) {
    mo_resid[m] = mo_resid[m] / 10;
  }
}
