data {
  int<lower=0> N; 
  int<lower=0> n_age; 
  int<lower=0> n_age_edu; 
  int<lower=0> n_edu; 
  int<lower=0> n_state; 
  int<lower=0> n_region; 
  int y[N];
  vector[N] female;
  vector[N] black;
  int age[N];
  int edu[N];
  int age_edu[N];
  int state[N];
  int region[n_state];
  vector[n_state] v_prev;
} 
parameters {
  vector[n_age_edu] b_age_edu;
  real b_0;
  real b_female;
  real b_black;
  real b_female_black;
  real mu;
  real b_v_prev_raw;
  real mu_age_edu;
  real<lower=0> sigma_age_raw;
  real<lower=0> sigma_edu_raw;
  real<lower=0> sigma_region_raw;
  real<lower=0> sigma_state_raw;
  real<lower=0> sigma_age_edu_raw;
  real<lower=0> sigma_beta;
  real<lower=0> xi_age;
  real<lower=0> xi_edu;
  real<lower=0> xi_age_edu;
  real<lower=0> xi_state;
  vector[n_age] b_age_raw;
  vector[n_edu] b_edu_raw;
  vector[n_state] b_state_raw;
  vector[n_region] b_region_raw;
  vector[4] beta;
} 
transformed parameters {
  vector[N] Xbeta;
  vector[n_age] b_age;
  vector[n_edu] b_edu;
  vector[n_region] b_region;
  vector[n_age_edu] b_age_edu_adj;
  vector[n_state] b_state_hat;
  vector[n_state] b_state;
  real mu_adj;
  real<lower=0> sigma_age;
  real<lower=0> sigma_edu;
  real<lower=0> sigma_age_edu;
  real<lower=0> sigma_state;
  real<lower=0> sigma_region;

  b_age <- xi_age * (b_age_raw - mean(b_age_raw));
  b_edu <- xi_edu * (b_edu_raw - mean(b_edu_raw));
  b_age_edu_adj <- b_age_edu - mean(b_age_edu);
  b_region <- xi_state * b_region_raw;
  b_state <- xi_state * (b_state_raw - mean(b_state_raw));
  mu_adj <- beta[1] + mean(b_age) + mean(b_edu) + mean(b_age_edu) +
     mean(b_state);

  sigma_age <- xi_age*sigma_age_raw;
  sigma_edu <- xi_edu*sigma_edu_raw;
  sigma_age_edu <- xi_age_edu*sigma_age_edu_raw;
  sigma_state <- xi_state*sigma_state_raw;
  sigma_region <- xi_state*sigma_region_raw;     # not "xi_region"

  for (i in 1:N)
    Xbeta[i] <- beta[1] + beta[2]*female[i] + beta[3]*black[i] +
      beta[4]*female[i]*black[i] +
      b_age[age[i]] + b_edu[edu[i]] + b_age_edu[age_edu[i]] +
      b_state[state[i]];
  for (j in 1:n_state)
    b_state_hat[j] <- b_region_raw[region[j]] + b_v_prev_raw*v_prev[j];
}
model {
  mu ~ normal (0, .0001);
  mu_age_edu ~ normal(0, .0001);
  sigma_age_raw ~ uniform(0, 100);
  sigma_edu_raw ~ uniform(0, 100);
  sigma_age_edu ~ uniform(0, 100);
  sigma_state_raw ~ uniform(0, 100);
  sigma_region_raw ~ uniform(0, 100);
  sigma_age_edu_raw ~ uniform(0, 100);
  sigma_beta ~ uniform(0, 100);

  b_age_raw ~ normal(0, sigma_age_raw);
  b_edu_raw ~ normal(0, sigma_edu_raw);
  b_age_edu ~ normal(mu_age_edu,sigma_age_edu);
  b_state_raw ~ normal(b_state_hat, sigma_state_raw);
  beta ~ normal(0, sigma_beta);

  b_v_prev_raw ~ normal(0, .0001);
  b_region_raw ~ normal(0, sigma_region_raw);

  xi_age ~ uniform (0, 100);
  xi_edu ~ uniform (0, 100);
  xi_age_edu ~ uniform (0, 100);
  xi_state ~ uniform (0, 100);

  y ~ bernoulli_logit(Xbeta);
}
