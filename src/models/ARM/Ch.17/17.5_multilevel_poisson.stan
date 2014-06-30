data {
  int<lower=0> N;
  int<lower=0> n_eth;
  int<lower=0> n_precint;

  int<lower=0, upper=n_precint> precint[N];
  int<lower=0, upper=n_eth> eth[N];
  vector[N] offeset;
  int<lower=0> stops[N];
}
parameters {
  real mu;
  real<lower=0> sigma_epsilon;
  real<lower=0> sigma_eth;
  real<lower=0> sigma_precint;

  vector[n_eth] b_eth;
  vector[n_precint] b_precint;
  vector[N] epsilon;
}
model {
  real mu_adj;
  vector[n_eth] b_eth_adj;
  vector[n_precint] b_precint_adj;
  vector[N] lambda;

  mu ~ normal(0, 100);
  mu_adj <- mu + mean(b_eth) + mean(b_precint);

  b_eth ~ normal(0, sigma_eth);
  b_eth_adj <- b_eth - mean(b_eth);

  b_precint ~ normal(0, sigma_precint);
  b_precint_adj <- b_precint - mean(b_precint);

  epsilon ~ normal(0, sigma_epsilon);

  for (i in 1:N)
    lambda[i] <- offeset[i] + mu + b_eth[eth[i]] + b_precint[precint[i]]
      + epsilon[i];

  stops ~ poisson_log(lambda);
}
