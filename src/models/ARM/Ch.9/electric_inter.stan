data {
  int<lower=0> N;
  vector[N] post_test;
  vector[N] treatment;
  vector[N] pre_test;
}
transformed data {
  vector[N] inter;           // interaction
  inter <- treatment .* pre_test;
}
parameters {
  vector[4] beta;
  real<lower=0> sigma;
}
model {
  post_test ~ normal(beta[1] + beta[2] * treatment + beta[3] * pre_test
                     + beta[4] * inter, sigma);
}
