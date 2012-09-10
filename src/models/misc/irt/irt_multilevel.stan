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
  real alpha[J];                 // ability of student j - mean ability
  real beta[K];                  // difficulty of question k
  real<lower=0> sigma_alpha;     // sd of student abilities  
  real <lower=0> sigma_beta;     // sd of question difficulties
}
model {
  alpha ~ normal(0,sigma_alpha); 
  beta ~ normal(0,sigma_beta);   
  delta ~ normal(.75,1);          // informative around known value
  for (n in 1:N)
    y[n] ~ bernoulli_logit(alpha[jj[n]] - beta[kk[n]] + delta);
}
