/**
 * Changepoint model for UK coal mining disasters between 1851 and
 * 1962.
 *
 * Data is a series D[t] for t in 1:T of yearly disaster counts for
 * year t.
 *
 * Priors:
 *   e ~ exponential(r_e)
 *   l ~ exponential(r_e)
 * 
 * Latent changepoint:
 *   s ~ uniform(1,T);
 *
 * Likelihood of disasters:
 *   D[t] ~ Poisson(t < s ? e : l)
 *
 * Estimand of interest is s, the point at which the disaster
 * rate changed.
 *
 * Marginalization for Stan implementation:
 *
 *   p(e,l|D) = exponential(e|r_e) * exponential(l|r_l)
 *            * SUM_s Uniform(s|1,T) * PROD_t Poisson(D[t] | t < s ? e : l)
 *
 * where
 *
 *   log SUM_s Uniform(s|1,T) * PROD_t Poisson(D[t] | t < s ? e : l)
 *   = LOG_SUM_EXP_s ( log Uniform(s|1,T) 
 *                     + SUM_t log Poisson(D[t] | t < s ? e : l) )
 *
 * Model from PyMC 2.3 documentation
 *   http://pymc-devs.github.io/pymc/index.html
 *   http://pymc-devs.github.io/pymc/tutorial.html#an-example-statistical-model
 * 
 * Data from: 
 *   R.G. Jarrett. A note on the intervals between coal mining
 *   disasters.  Biometrika, 66:191–193, 1979.
 */
data {
  real<lower=0> r_e;  // before change disaster rate prior
  real<lower=0> r_l;  // after change disaster rate prior

  int<lower=1> T;     // years
  int<lower=0> D[T];  // disasters per year
}
transformed data {
  real log_unif;
  log_unif <- -log(T);  // log p(s)
}
parameters {
  real<lower=0> e;    // before change disaster rate
  real<lower=0> l;    // after change disaster rate
}
transformed parameters {
  vector[T] lp;
  lp <- rep_vector(log_unif, T);
  for (s in 1:T)
    for (t in 1:T)
      lp[s] <- lp[s] + poisson_log(D[t], if_else(t < s, e, l));
}
model {
  // prior
  e ~ exponential(r_e);
  l ~ exponential(r_l);

  // likelihood
  increment_log_prob(log_sum_exp(lp));
}    
generated quantities {
  int<lower=1,upper=T> s;
  s <- categorical_rng(softmax(lp));
}
