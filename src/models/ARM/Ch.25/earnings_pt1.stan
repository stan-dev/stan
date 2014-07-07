data {
  int<lower=0> N; 
  vector[N] any_charity;
  vector[N] any_ssi;
  vector[N] any_welfare;
  int<lower=0,upper=1> earnings[N];
  vector[N] educ_r;
  vector[N] immig;
  vector[N] male;
  vector[N] over65;
  vector[N] white;
}
parameters {
  vector[9] beta;
} 
model {
  earnings ~ bernoulli_logit(beta[1] + beta[2] * male + beta[3] * over65 
                             + beta[4] * white + beta[5] * immig 
                             + beta[6] * educ_r + beta[7] * any_ssi 
                             + beta[8] * any_welfare + beta[9] * any_charity);
}
