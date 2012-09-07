data {
  real U;
  int<lower=0> N_censored;
  int<lower=0> N_observed;
  real<upper=U> y[N_observed];
}
parameters { 
  real mu;
}
model {
  for (n in 1:N_observed)
    y[n] ~ normal(mu,1.0) T[,U];
  lp__ <- lp__ + N_censored * log1m(normal_cdf(U,mu,1.0));
}


