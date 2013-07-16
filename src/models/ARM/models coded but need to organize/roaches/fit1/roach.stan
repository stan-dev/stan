data {
  int<lower=0> N; 
  vector[N] roach1;
  vector[N] treatment;
  vector[N] senior;
  vector[N] exposure2;
  int<lower=0> y[N];
}
parameters {
  vector[4] beta;
} 
model {
  for (n in 1:N)
    y[n] ~ poisson(exposure2[n] * exp(beta[1] + beta[2] * roach1[n] 
                                      + beta[3] * treatment[n]
                                      + beta[4] * senior[n]));
}
