/**
 * HIERARCHICAL PL2
 */

data {
  int<lower=1> J;                // number of students
  int<lower=1> K;                // number of questions
  int<lower=1> N;                // number of observations
  int<lower=1,upper=J> jj[N];    // student for observation n
  int<lower=1,upper=K> kk[N];    // question for observation n
  int<lower=0,upper=1> y[N];     // correctness of observation n
}
parameters {    
  real delta;                    // mean student ability
  real alpha[J];                 // ability for student j - mean ability
  real beta[K];                  // difficulty for question k
  real log_gamma[K];             // discriminativeness for question k
  real<lower=0> sigma_alpha;     // sd of student abilities  
  real<lower=0> sigma_beta;      // sd of question difficulties 
  real<lower=0> sigma_gamma;     // sd of log question discriminativeness
}
model {
  alpha ~ normal(0,sigma_alpha); 
  beta ~ normal(0,sigma_beta);   
  log_gamma ~ normal(0,sigma_gamma);
  delta ~ cauchy(0,5);
  sigma_alpha ~ cauchy(0,5);
  sigma_beta ~ cauchy(0,5);
  sigma_gamma ~ cauchy(0,5);
  for (n in 1:N)
    y[n] ~ bernoulli_logit( exp(log_gamma[kk[n]])
                            * (alpha[jj[n]] - beta[kk[n]] + delta) );
}
