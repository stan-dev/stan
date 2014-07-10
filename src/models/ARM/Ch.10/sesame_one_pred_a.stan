data {
  int<lower=0> N; 
  vector[N] encouraged;
  vector[N] watched;
}
parameters {
  vector[2] beta;
  real<lower=0> sigma;
} 
model {
  watched ~ normal(beta[1] + beta[2] * encouraged,sigma);
}
