data {
  int<lower=0> N; 
  vector[N] pre_test;
  vector[N] post_test;
  vector[N] treatment;
}
parameters {
  vector[3] beta;
  real<lower=0> sigma;
} 
model {
  post_test ~ normal(beta[1] + beta[2] * treatment + beta[3] * pre_test,sigma);
}
