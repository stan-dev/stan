data {
  int<lower=0> N; 
  int<lower=0> n_age; 
  int<lower=0> n_edu; 
  int<lower=0> n_state; 
  int<lower=0> n_region; 
  vector[N] y;
  vector[N] female;
  vector[N] black;
  vector[N] age;
  vector[N] edu;
  vector[N] state;
  vector[N] region;
  vector[N] v_prev;
} 
parameters {
  matrix[n_age,n_edu] b_age_edu;
  real b_0;
  real b_female;
  real b_black;
  real b_female_black;
  real mu;
  real b_v_prev_raw;
  real mu_age_edu;
  real sigma_age_raw;
  real sigma_edu_raw;
  real sigma_age_edu;
  real sigma_region_raw;
  real sigma_state_raw;
  real xi_age;
  real xi_edu;
  real xi_age_edu;
  real xi_state;
  vector[n_age] b_age_raw;
  vector[n_edu] b_edu_raw;
  vector[n_state] b_state_raw;
  vector[n_region] b_region;
} 
transformed paramaters {
  vector[N] Xbeta;
  vector[N] p;
  vector[N] p_bound;
  vector[n_age] b_age;
  vector[n_edu] b_edu;
  vector[n_region] b_region;
  matrix[n_age,n_edu] b_age_edu_adj;
  vector[n_state] b_state_raw_hat;
  vector[n_state] b_state;
  real mu_adj;

  b_age <- xi_age * (b_age_raw - mean(b_age_raw))
  b_edu <- xi_edu * (b_edu_raw - mean(b_edu_raw));
  b_age_edu_adj <- b_age_edu - mean(b_age_edu);
  b_region <- xi_state * b_region_raw;
  b_state <- xi_state * (b_state_raw - mean(b_state_raw));
  mu_adj <- b_0 + mean(b_age) + mean(b_edu) + mean(b_age_edu) +
     mean(b_state);

  for (i in 1:N){
    Xbeta[i] <- b.0 + b_female*female[i] + b_black*black[i] +
      b_female_black*female[i]*black[i] +
      b_age[age[i]] + b_edu[edu[i]] + b_age_edu[age[i],edu[i]] +
      b_state[state[i]];
    p[i] <- inv_logit(Xbeta[i]);
    p_bound[i] <- max(0,min(1,p[i]));
  }
  for (j in 1:n.state)
    b_state_hat[j] <- b_region_raw[region[j]] + b_v_prev_raw*v_prev[j];
}
model {

  y ~ binomial(p_bound,1)
  
  b_0 ~ dnorm (0, .0001);
  b_female ~ dnorm (0, .0001);
  b_black ~ dnorm (0, .0001);
  b_female_black ~ dnorm (0, .0001);

  mu ~ dnorm (0, .0001);

  b_age_raw ~ normal(0, sigma_age_raw);
  b_edu_raw ~ normal(0, sigma_edu_raw);
  b_age_edu ~ normal(mu_age_edu,sigma_age_edu);
  b_state_raw ~ normal(b_state_hat, sigma_state_raw);

  b_v_prev_raw ~ normal(0, .0001);
  b_region_raw ~ normal(0, sigma_region_raw);

  mu_age_edu ~ normal(0, .0001);
  sigma_age_raw ~ uniform(0, 100);
  sigma_edu_raw ~ uniform(0, 100);
  sigma_age_edu ~ uniform(0, 100);
  sigma_state_raw ~ uniform(0, 100);
  sigma_region_raw ~ uniform(0, 100);

  xi_age ~ uniform (0, 100)
  xi_edu ~ uniform (0, 100)
  xi_age_edu ~ uniform (0, 100)
  xi_state ~ uniform (0, 100)
}
generated quantities {
  real sigma_age;
  real sigma_edu;
  real sigma_age_edu;
  real sigma_age_state;
  real sigma_age_region;

  sigma_age <- xi_age*sigma_age_raw
  sigma_edu <- xi_edu*sigma_edu_raw
  sigma_age_edu <- xi_age_edu*sigma_age_edu_raw
  sigma_state <- xi_state*sigma_state_raw
  sigma_region <- xi_state*sigma_region_raw     # not "xi_region"
}