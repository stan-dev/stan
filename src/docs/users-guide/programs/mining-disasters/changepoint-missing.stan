// this is a trivial missing data problem because
// there are only two possible cases for missingness,
// and in both cases, the posterior predictive is
// analytic given e and l

// but it does make sense to do the model with not every
// year observed, assuming unobserved years are missing at
// random

data {
  int<lower=0> t_l;                         // earliest time
  int<lower=t_l> t_h;                       // latest time

  real<lower=0> r_e;                        // before rate prior
  real<lower=0> r_l;                        // after rate prior

  int<lower=1> N_obs;                       // num observed
  int<lower=t_l,upper=t_h> T_obs[N_obs];    // times
  int<lower=0> D_obs[N_obs];                // disasters

  int<lower=0> N_miss;                      // num missing
  int<lower=t_l,upper=t_h> T_miss[N_miss];  // missing times
}
transformed data {
  real log_unif;
  int<lower=0> S;
  S <- t_h - t_l + 1;
  log_unif <- -log(S);  // log p(s) = log Uniform(s|1,T)
}
parameters {
  real<lower=0> e;    // before rate
  real<lower=0> l;    // after rate
}
transformed parameters {
  real lp[S];
  lp <- rep_array(log_unif, T);
  for (s in t_l:t_h)
    for (n in 1:N)
      lp[s] <- lp[s] + poisson_log(D[n], if_else(T[n] < s, e, l));
}
model {
  // prior
  e ~ exponential(r_e);
  l ~ exponential(r_l);

  // likelihood
  increment_log_prob(log_sum_exp(lp));
}
generated quantities {
  int<lower=0> D_miss[N_miss];
  for (n in 1:N_miss)
    D_miss[n] <- poisson_rng(if_else(T_miss[n] < s, e, l));
}
