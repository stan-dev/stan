data {
  int<lower=0> N; 
  int<lower=0> n_age; 
  int<lower=0> n_edu; 
  int<lower=0> n_age_edu; 
  int<lower=0> n_state; 
  int<lower=0> n_region; 
  int y[N];
  vector[N] female;
  vector[N] black;
  int age[N];
  int edu[N];
  int state[N];
  int region[n_state];
  vector[n_state] v_prev;
  int age_edu[N];
} 
parameters {
  real mu;
  real b_v_prev;
  real mu_age;
  real mu_edu;
  real mu_age_edu;
  real mu_region;
  real<lower=0> sigma_age;
  real<lower=0> sigma_edu;
  real<lower=0> sigma_age_edu;
  real<lower=0> sigma_region;
  real<lower=0> sigma_state;
  vector[n_age] b_age;
  vector[n_edu] b_edu;
  vector[n_state] b_state;
  vector[n_region] b_region;
  vector[n_age_edu] b_age_edu;
  vector[4] beta;
} 
transformed parameters {
  vector[N] Xbeta;
  vector[N] p;
  vector[N] p_bound;
  vector[n_age] b_age_adj;
  vector[n_edu] b_edu_adj;
  vector[n_age_edu] b_age_edu_adj;
  vector[n_state] b_state_hat;
  vector[n_region] b_region_adj;
  real mu_adj;
  for (i in 1:N)
    Xbeta[i] <- beta[1] + beta[2]*female[i] + beta[3]*black[i] +
      beta[4]*female[i]*black[i] +
      b_age[age[i]] + b_edu[edu[i]] + b_age_edu[age_edu[i]] +
      b_state[state[i]];

  mu_adj <- beta[1] + mean(b_age) + mean(b_edu) + mean(b_age_edu) +
     mean(b_state);
  b_age_adj <- b_age - mean(b_age);
  b_edu_adj <- b_edu - mean(b_edu);
  b_age_edu_adj <- b_age_edu - mean(b_age_edu);
  b_region_adj <- b_region - mean(b_region);

  for (j in 1:n_state)
    b_state_hat[j] <- b_region[region[j]] + b_v_prev*v_prev[j];
}
model {  
  mu_age ~ normal(0, 100);
  mu_edu ~ normal(0, 100);
  mu_age_edu ~ normal(0, 100);
  mu_region ~ normal(0, 100);
  mu ~ normal(0, 100);
  sigma_age ~ uniform(0, 100);
  sigma_edu ~ uniform(0, 100);
  sigma_age_edu ~ uniform(0, 100);
  sigma_region ~ uniform(0, 100);
  sigma_state ~ uniform(0, 100);

  beta ~ normal(0, 100);
  b_age ~ normal(mu_age, sigma_age);
  b_edu ~ normal(mu_edu, sigma_edu);
  b_age_edu ~ normal(mu_age_edu,sigma_age_edu);
  b_state ~ normal(b_state_hat, sigma_state);

  b_v_prev ~ normal(0, 100);
  b_region ~ normal(mu_region, sigma_region);

  y ~ bernoulli_logit(Xbeta);
}
