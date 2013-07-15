data {
  int<lower=0> N; 
  vector[N] encouraged;
  int site[N];
  vector[N] setting;
  vector[N] pretest;
  vector[N] watched;
}
transformed data {
  vector[N] site2; 
  vector[N] site3; 
  vector[N] site4; 
  vector[N] site5; 
  for (i in 1:N) {
    if (site[i] == 2)
      site2[i] <- 1;
    if (site[i] == 3)
      site3[i] <- 1;
    if (site[i] == 4)
      site4[i] <- 1;
    if (site[i] == 5)
      site5[i] <- 1;
  }
}
parameters {
  vector[8] beta;
  real<lower=0> sigma;
} 
model {
  watched ~ normal(beta[1] + beta[2] * encouraged + beta[3] * pretest 
             + beta[4] * site2 + beta[5] * site3 + beta[6] * site4 
             + beta[7] * site5 + beta[8] * setting,sigma);
}
