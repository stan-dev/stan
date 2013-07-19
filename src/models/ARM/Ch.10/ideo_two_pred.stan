data {
  int<lower=0> N; 
  vector[N] score1;
  vector[N] x;
  vector[N] party;
}
parameters {
  vector[3] beta;
  real<lower=0> sigma;
} 
model {
  score1 ~ normal(beta[1] + beta[2] * party + beta[3] * x,sigma);
}
