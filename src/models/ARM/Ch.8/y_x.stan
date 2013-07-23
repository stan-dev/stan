data {
  int<lower=0> N; 
  vector[N] y;
  vector[N] x;
}
parameters {
  vector[2] beta;
  real<lower=0> sigma;
} 
model {
  y ~ normal(beta[1] + beta[2] * x,sigma);
}
