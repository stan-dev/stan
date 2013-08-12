data {
  int<lower=0> N;
  vector[N] post_test;
  vector[N] supp;
  vector[N] pre_test;
}
parameters {
  vector[3] beta;
  real<lower=0> sigma;
}
model {
  post_test ~ normal(beta[1] + beta[2] * supp + beta[3] * pre_test, sigma);
}
