data {
  int<lower=0> N; 
  int<lower=0> n_age; 
  int<lower=0> n_age_edu; 
  int<lower=0> n_edu; 
  int<lower=0> n_region; 
  int<lower=0> n_state; 
  int<lower=1,upper=n_age> age[N];
  int<lower=1,upper=n_age_edu> age_edu[N];
  vector<lower=0,upper=1>[N] black;
  int<lower=1,upper=n_edu> edu[N];
  vector<lower=0,upper=1>[N] female;
  int<lower=1,upper=n_region> region[n_state];
  int<lower=1,upper=n_state> state[N];
  vector[n_state] v_prev;
  int<lower=0,upper=1> y[N];
} 
parameters {
  vector[n_age] b_age;
  vector[n_age_edu] b_age_edu;
  vector[n_edu] b_edu;
  vector[n_region] b_region;
  vector[n_state] b_state;
  real b_v_prev;
  vector[4] beta;
  real mu;
  real mu_age;
  real mu_age_edu;
  real mu_edu;
  real mu_region;
  real<lower=0,upper=100> sigma_age;
  real<lower=0,upper=100> sigma_edu;
  real<lower=0,upper=100> sigma_age_edu;
  real<lower=0,upper=100> sigma_region;
  real<lower=0,upper=100> sigma_state;
} 
transformed parameters {
  vector[n_age] b_age_adj;
  vector[n_age_edu] b_age_edu_adj;
  vector[n_edu] b_edu_adj;
  vector[n_region] b_region_adj;
  vector[n_state] b_state_hat;
  real mu_adj;
  vector[N] Xbeta;
  vector[N] p;
  vector[N] p_bound;

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
    b_state_hat[j] <- b_region[region[j]] + 100 * b_v_prev * v_prev[j];
}
model {  
  mu_age ~ normal(0, 1);
  mu_edu ~ normal(0, 1);
  mu_age_edu ~ normal(0, 1);
  mu_region ~ normal(0, 1);
  mu ~ normal(0, 100);
  sigma_age ~ uniform(0, 100);
  sigma_edu ~ uniform(0, 100);
  sigma_age_edu ~ uniform(0, 100);
  sigma_region ~ uniform(0, 100);
  sigma_state ~ uniform(0, 100);

  beta ~ normal(0, 100);
  b_age ~ normal(100 * mu_age, sigma_age);
  b_edu ~ normal(100 * mu_edu, sigma_edu);
  b_age_edu ~ normal(100 * mu_age_edu,sigma_age_edu);
  b_state ~ normal(b_state_hat, sigma_state);

  b_v_prev ~ normal(0, 1);
  b_region ~ normal(100 * mu_region, sigma_region);

  y ~ bernoulli_logit(Xbeta);
}
