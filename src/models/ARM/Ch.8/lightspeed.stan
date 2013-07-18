data {
  int<lower=0> N; 
  vector[N] y;
}
parameters {
  vector[1] beta;
  real<lower=0> sigma;
} 
model {
  y ~ normal(beta[1],sigma);
}
