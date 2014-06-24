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
  vector[N] lambda;
  real<lower=0> tau;
} 
transformed parameters {
  real<lower=0> sigma;

  sigma <- 1.0 / sqrt(tau);
}
model {
  tau ~ gamma(0.001, 0.001);
  for (i in 1:N) {
    lambda[i] ~ normal(0, sigma);
    y[i] ~ poisson_log(lambda[i] + log_expo[i] + beta[1] + beta[2]*roach1[i] 
                       + beta[3]*senior[i] + beta[4]*treatment[i]);
  }
}
