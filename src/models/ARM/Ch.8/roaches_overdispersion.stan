data {
  int<lower=0> N; 
  vector[N] exposure2;
  vector[N] roach1;
  vector[N] senior;
  vector[N] treatment;
  int y[N];
}
transformed data {
  vector[N] log_expo;

  log_expo <- log(exposure2);
}
parameters {
  vector[4] beta;
  real<lower=0> omega;
} 
model {
  for (n in 1:N)
    y[n] ~ neg_binomial(exposure2[n] * exp(beta[1] + beta[2] * roach1[n] 
                        + beta[3] * treatment[n] + beta[4] * senior[n]) 
                        / (omega - 1),1.0 / (omega - 1));
}
