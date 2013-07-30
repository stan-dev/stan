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
  real b_v_prev;
  real mu_age;
  real mu_edu;
  real mu_age_edu;
  real mu_region;
  real sigma_age;
  real sigma_edu;
  real sigma_age_edu;
  real sigma_region;
  real sigma_state;
  vector[n_age] b_age;
  vector[n_edu] b_edu;
  vector[n_state] b_state;
  vector[n_region] b_region;
} 
transformed paramaters {
  vector[N] Xbeta;
  vector[N] p;
  vector[N] p_bound;
  vector[n_age] b_age_adj;
  vector[n_edu] b_edu_adj;
  matrix[n_age,n_edu] b_age_edu_adj;
  vector[n_state] b_state_hat;
  real mu_adj;
  for (i in 1:N){
    Xbeta[i] <- b.0 + b_female*female[i] + b_black*black[i] +
      b_female_black*female[i]*black[i] +
      b_age[age[i]] + b_edu[edu[i]] + b_age_edu[age[i],edu[i]] +
      b_state[state[i]];
    p[i] <- inv_logit(Xbeta[i]);
    p_bound[i] <- max(0,min(1,p[i]));
  }

  mu_adj <- b_0 + mean(b_age) + mean(b_edu) + mean(b_age_edu) +
     mean(b_state);
  b_age_adj <- b_age - mean(b_age);
  b_edu_adj <- b_edu - mean(b_edu);
  b_age_edu_adj <- b_age_edu - mean(b_age_edu);
  b_region_adj <- b_region - mean(b_region);

  for (j in 1:n.state)
    b_state_hat[j] <- b_region[region[j]] + b_v_prev*v_prev[j];
}
model {

  y ~ binomial(p_bound,1)
  
  b_0 ~ dnorm (0, .0001);
  b_female ~ dnorm (0, .0001);
  b_black ~ dnorm (0, .0001);
  b_female_black ~ dnorm (0, .0001);

  mu ~ dnorm (0, .0001);

  b_age ~ normal(mu_age, sigma_age);
  b_edu ~ normal(mu_edu, sigma_edu);
  b_age_edu ~ normal(mu_age_edu,sigma_age_edu);
  b_state ~ normal(mu_state, sigma_state);

  b_v_prev ~ normal(0, .0001);
  b_region ~ normal(mu_region, sigma_region);

  mu_age ~ normal(0, .0001);
  mu_edu ~ normal(0, .0001);
  mu_age_edu ~ normal(0, .0001);
  mu_region ~ normal(0, .0001);
  sigma_age ~ uniform(0, 100);
  sigma_edu ~ uniform(0, 100);
  sigma_age_edu ~ uniform(0, 100);
  sigma_state ~ uniform(0, 100);
  sigma_region ~ uniform(0, 100);
}
