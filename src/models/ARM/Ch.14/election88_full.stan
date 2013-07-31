data {
  int<lower=0> N; 
  int<lower=0> n_age; 
  int<lower=0> n_edu; 
  int<lower=0> n_age_edu; 
  int<lower=0> n_region_full; 
  int<lower=0> n_state; 
  int y[N];
  vector[N] black;
  vector[N] female;
  vector[N] v_prev_full;
  int age[N];
  int age_edu[N];
  int region_full[N];
  int state[N];
  int edu[N];
} 
parameters {
  vector[n_age] a;
  vector[n_edu] b;
  vector[n_age_edu] c;
  vector[n_state] d;
  vector[n_region_full] e;
  vector[5] beta;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;
  real<lower=0> sigma_c;
  real<lower=0> sigma_d;
  real<lower=0> sigma_e;
  real<lower=0> sigma_beta;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] <- beta[1] + beta[2] * black[i] + beta[3] * female[i]
                + beta[5] * female[i] * black[i] 
                + beta[4] * v_prev_full[i] + a[age[i]] + b[edu[i]] 
                + c[age_edu[i]] + d[state[i]] + e[region_full[i]];
} 
model {
  sigma_a ~ uniform(0, 100);
  a ~ normal (0, sigma_a);

  sigma_b ~ uniform(0, 100);
  b ~ normal (0, sigma_b);

  sigma_c ~ uniform(0, 100);
  c ~ normal (0, sigma_c);

  sigma_d ~ uniform(0, 100);
  d ~ normal (0, sigma_d);

  sigma_e ~ uniform(0, 100);
  e ~ normal (0, sigma_e);

  sigma_beta ~ uniform(0, 100);
  beta ~ normal(0, sigma_beta);

  y ~ bernoulli_logit(y_hat);
}
