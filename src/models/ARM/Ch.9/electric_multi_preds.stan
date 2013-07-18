data {
  int<lower=0> N; 
  vector[N] post_test;
  vector[N] pre_test;
  vector[N] treatment;
}
parameters {
  vector[3] beta1;
  real<lower=0> sigma1;
} 
model {
  post_test ~ normal(beta1[1] + beta1[2] * treatment + beta1[3] * pre_test,sigma1);
}
