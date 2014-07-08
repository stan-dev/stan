data {
  int<lower=0> N;
  vector[N] ppvt;
  vector[N] hs;
  vector[N] afqt;
}
parameters {
  vector[3] beta;
  real<lower=0> sigma;
}
model {
  ppvt ~ normal(beta[1] + beta[2] * hs + beta[3] * afqt, sigma);
}
