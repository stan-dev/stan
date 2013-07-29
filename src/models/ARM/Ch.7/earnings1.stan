data {
  int<lower=0> N; 
  int<lower=0,upper=1> earn_pos[N];
  vector[N] height;
  vector[N] male;
} 
parameters {
  vector[3] beta;
  real<lower=0> sigma;
} 
model {
  earn_pos ~ bernoulli_logit(beta[1] + beta[2] * height + beta[3] * male);
}
