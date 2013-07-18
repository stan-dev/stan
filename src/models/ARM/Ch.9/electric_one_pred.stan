data {
  int<lower=0> N; 
  vector[N] post_test;
  vector[N] treatment;
}
parameters {
  vector[2] beta1;
  real<lower=0> sigma1;
} 
model {
  post_test ~ normal(beta1[1] + beta1[2] * treatment,sigma1);
}
