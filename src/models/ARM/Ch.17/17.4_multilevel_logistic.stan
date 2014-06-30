data {
  int<lower=0> N;
  int<lower=0> n_age;
  int<lower=0> n_edu;
  int<lower=0> n_region;
  int<lower=0> n_state;

  int<lower=0, upper=1> female[N];
  int<lower=0, upper=1> black[N];
  int<lower=0, upper=n_age> age[N];
  int<lower=0, upper=n_edu> edu[N];
  int<lower=0, upper=n_state> region[n_state];
  int<lower=0, upper=n_state> state[N];
  int<lower=0, upper=1> y[N];
  vector[n_state] v_prev;
}
parameters {
  real<lower=0> sigma;
  real<lower=0> sigma_age;
  real<lower=0> sigma_edu;
  real<lower=0> sigma_state;
  real<lower=0> sigma_region;
  real<lower=0> sigma_age_edu;

  real b_0;
  real b_female;
  real b_black;
  real b_female_black;

  real b_v_prev;

  vector[n_age] b_age;
  vector[n_edu] b_edu;
  vector[n_region] b_region;
  matrix[n_age,n_edu] b_age_edu;

  vector[n_state] b_hat;
}
model {
  vector[N] p;
  vector[n_state] b_state_hat;

  b_0 ~ normal(0, 100);
  b_female ~ normal(0, 100);
  b_black ~ normal(0, 100);
  b_female_black ~ normal(0, 100);

  b_age ~ normal(0, sigma_age);
  b_edu ~ normal(0, sigma_edu);
  b_region ~ normal(0, sigma_region);

  for (j in 1:n_age) {
    for (i in 1:n_edu)
      b_age_edu[j,i] ~ normal(0, sigma_age_edu);
  }

  b_v_prev ~ normal(0, 100);

  for (j in 1:n_state)
    b_state_hat[j] <- b_region[region[j]] + b_v_prev * v_prev[j];

  b_hat ~ normal(b_state_hat, sigma_state);

  for (i in 1:N)
    p[i] <- fmax(0, fmin(1, inv_logit(b_0 + b_female*female[i] 
      + b_black*black[i] + b_female_black*female[i]*black[i] +
      b_age[age[i]] + b_edu[edu[i]] + b_age_edu[age[i],edu[i]] +
      b_hat[state[i]])));

  y ~ bernoulli(p);
}
