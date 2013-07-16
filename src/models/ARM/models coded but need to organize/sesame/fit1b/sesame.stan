data {
  int<lower=0> N; 
  vector[N] encouraged;
  vector[N] y;
}
parameters {
  vector[2] beta;
  real<lower=0> sigma;
} 
model {
  y ~ normal(beta[1] + beta[2] * encouraged,sigma);
}
