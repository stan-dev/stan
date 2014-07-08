data {
  int<lower=0> N;
  vector[N] kid_score;
  vector[N] mom_hs;
}
parameters {
  vector[2] beta;
  real<lower=0> sigma;
}
model {
  kid_score ~ normal(beta[1] + beta[2] * mom_hs, sigma);
}
