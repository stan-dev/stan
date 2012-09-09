# http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol1.pdf
# Page 48: Mice Weibull regression 
# http://www.openbugs.info/Examples/Mice.html

# note that Stan and JAGS have different parameterization for Weibull
# distribution

# Data is transformed using data_reorg.R 
data {
  int<lower=0> N_uncensored;
  int<lower=0> N_censored;
  int<lower=0> M;
  int<lower=1,upper=M> group_uncensored[N_uncensored];
  int<lower=1,upper=M> group_censored[N_censored];
  real<lower=0> censor_time[N_censored];
  real<lower=0> t_uncensored[N_uncensored];
}

parameters {
  real<lower=0> r;
  real beta[M];
  real<lower=1> t2_censored[N_censored]; # t_censored / censor_time 
}

model {
  r ~ exponential(0.001);
  beta ~ normal(0, 100);
  for (n in 1:N_uncensored) {
    t_uncensored[n] ~ weibull(r, exp(-beta[group_uncensored[n]] / r));
  }
  for (n in 1:N_censored) {
    t2_censored[n] ~ weibull(r, exp(-beta[group_censored[n]] / r) / censor_time[n]);
  }
}

generated quantities {
  real median[M];
  real pos_control;
  real test_sub;
  real veh_control;
  
  for (m in 1:M)
    median[m] <- pow(log(2) * exp(-beta[m]), 1/r);
  
  veh_control <- beta[2] - beta[1];
  test_sub    <- beta[3] - beta[1];
  pos_control <- beta[4] - beta[1];
}
