data {
  int<lower=0> N; 
  vector[N] ppvt;
  vector[N] afqt;
  vector[N] hs;
} 
transformed data {
  vector[N] c_afqt;
  real mu_afqt;
  real sd_afqt;

  mu_afqt <- mean(afqt);
  sd_afqt <- sd(afqt);
  c_afqt <- (afqt - mu_afqt) * 15.0 / sd_afqt + 100.0;
}
parameters {
  vector[3] beta;
  real<lower=0> sigma;
} 
model { 
  ppvt ~ normal(beta[1] + beta[2] * hs + beta[3] * afqt, sigma);
}
