data {
  int<lower=0> N; 
  vector[N] exposure2;
  int y[N];
  vector[N] roach1;
  vector[N] treatment;
  vector[N] senior;
}
parameters {
  vector[4] beta;
} 
model {
  y ~ poisson_log(log(exposure2) + beta[1] + beta[2] * roach1 + beta[3] * treatment
                      + beta[4] * senior);
}
