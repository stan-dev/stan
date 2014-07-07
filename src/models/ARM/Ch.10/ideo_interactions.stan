data {
  int<lower=0> N; 
  vector[N] party;
  vector[N] score1;
  vector[N] x;
}
transformed data {
  vector[N] inter;

  inter <- party .* x;
}
parameters {
  vector[4] beta;
  real<lower=0> sigma;
} 
model {
  score1 ~ normal(beta[1] + beta[2] * party + beta[3] * x + beta[4] * inter,sigma);
}
