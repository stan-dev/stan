data {
  int<lower=0> N; 
  vector[N] pretest;
  vector[N] setting;
  int site[N];
  vector[N] watched_hat;
  vector[N] y;
}
transformed data {
  vector[N] site2; 
  vector[N] site3; 
  vector[N] site4; 
  vector[N] site5; 
  for (i in 1:N) {
    site2[i] <- site[i] == 2;
    site3[i] <- site[i] == 3;
    site4[i] <- site[i] == 4;
    site5[i] <- site[i] == 5;
  }
}
parameters {
  vector[8] beta;
  real<lower=0> sigma;
} 
model {
  y ~ normal(beta[1] + beta[2] * watched_hat + beta[3] * pretest 
             + beta[4] * site2 + beta[5] * site3 + beta[6] * site4 
             + beta[7] * site5 + beta[8] * setting,sigma);
}
