data {
  int<lower=0> N; 
  vector[N] party;
  vector[N] score1;
  vector[N] z1; //z value for party 0, 0 otherwise 
  vector[N] z2; //z value for party 1, 0 otherwise 
}
parameters {
  vector[4] beta;
  real<lower=0> sigma;
} 
model {
  score1 ~ normal(beta[1] + beta[2] * party + beta[3] * z1 + beta[4] * z2,sigma);
}
