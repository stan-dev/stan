data {
  int<lower=0> N; 
  vector[N] any_charity;
  vector[N] any_ssi;
  vector[N] any_welfare;
  vector[N] earnings;
  vector[N] educ_r;
  vector[N] immig;
  vector[N] interest;
  vector[N] male;
  vector[N] over65;
  vector[N] white;
  vector[N] workhrs_top;
  vector[N] workmos;
}
parameters {
  vector[12] beta;
  real<lower=0> sigma;
} 
model {
  earnings ~ normal(beta[1] + beta[2] * interest + beta[3] * male 
                    + beta[4] * over65 + beta[5] * white
                    + beta[6] * immig + beta[7] * educ_r + beta[8] * workmos 
                    + beta[9] * workhrs_top + beta[10] * any_ssi 
                    + beta[11] * any_welfare + beta[12] * any_charity,sigma);
}
