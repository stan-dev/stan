#DOC See \url{http://www.openbugs.info/Examples/Blockers.html}.
data {
  int<lower=0> N; 
  int<lower=0> nt[N]; 
  int<lower=0> rt[N]; 
  int<lower=0> nc[N]; 
  int<lower=0> rc[N]; 
} 
parameters {
  real d; 
  real<lower=0> sigmasq_delta; 
  vector[N] mu;
  vector[N] delta;
  real delta_new;
} 
transformed parameters {
  real<lower=0> sigma_delta; 
  sigma_delta <- sqrt(sigmasq_delta); 
} 
model {
  rt ~ binomial_logit(nt, mu + delta);
  rc ~ binomial_logit(nc, mu);
  delta  ~ student_t(4, d, sigma_delta); 
  mu ~ normal(0, sqrt(1E5));
  d ~ normal(0, 1E3); 
  sigmasq_delta ~ inv_gamma(1E-3, 1E-3); 
  // FIXME: sample in generated quantities in later version
  delta_new ~ student_t(4, d, sigma_delta); 
}
