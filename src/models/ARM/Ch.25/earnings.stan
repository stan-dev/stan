data {
  int<lower=0> N; 
  vector[N] any_charity;
  vector[N] any_ssi;
  vector[N] any_welfare;
  vector[N] earnings;
  vector[N] educ_r;
  vector[N] immig;
  vector[N] male;
  vector[N] over65;
  vector[N] white;
  vector[N] workhrs_top;
  vector[N] workmos;
}
parameters {
  vector[11] beta;
  real<lower=0> sigma;
} 
model {
  earnings ~ normal(beta[1] + beta[2] * male + beta[3] * over65 + beta[4] * white
                    + beta[5] * immig + beta[6] * educ_r + beta[7] * workmos 
                    + beta[8] * workhrs_top + beta[9] * any_ssi 
                    + beta[10] * any_welfare + beta[11] * any_charity,sigma);
}
