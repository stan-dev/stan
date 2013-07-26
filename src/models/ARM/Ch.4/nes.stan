data {
  int<lower=0> N; 
  vector[N] partyid7;
  vector[N] real_ideo;
  vector[N] race_adj;
  vector[N] educ1;
  vector[N] gender;
  vector[N] income;
  int age[N];
}
transformed data {
  matrix[N,3] fact_age;
  for (n in 1:N)
    if (age[n] > 1)
      fact_age[n,age[n]-1] <- 1;
}
parameters {
  vector[6] beta;
  vector[3] beta_fact;
  real<lower=0> sigma;
} 
model {
  partyid7 ~ normal(beta[1] + beta[2] * real_ideo + beta[3] * race_adj + beta[4] * educ1 + beta[5] * gender + beta[6] * income + fact_age * beta_fact, sigma);
}
