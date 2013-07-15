data {
  int<lower=0> N1; 
  int<lower=0> N2; 
  int<lower=0> N3; 
  int<lower=0> N4; 
  vector[N1] post_test1;
  vector[N2] post_test2;
  vector[N3] post_test3;
  vector[N4] post_test4;
  vector[N1] pre_test1;
  vector[N2] pre_test2;
  vector[N3] pre_test3;
  vector[N4] pre_test4;
  vector[N1] supp1;
  vector[N2] supp2;
  vector[N3] supp3;
  vector[N4] supp4;
}
parameters {
  vector[3] beta1;
  vector[3] beta2;
  vector[3] beta3;
  vector[3] beta4;
  real<lower=0> sigma1;
  real<lower=0> sigma2;
  real<lower=0> sigma3;
  real<lower=0> sigma4;
} 
model {
  post_test1 ~ normal(beta1[1] + beta1[2] * supp1 + beta1[3] * pre_test1,sigma1);
  post_test2 ~ normal(beta2[1] + beta2[2] * supp2 + beta2[3] * pre_test2,sigma2);
  post_test3 ~ normal(beta3[1] + beta3[2] * supp3 + beta3[3] * pre_test3,sigma3);
  post_test4 ~ normal(beta4[1] + beta4[2] * supp4 + beta4[3] * pre_test4,sigma4);
}
