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
  vector[N1] treatment1;
  vector[N2] treatment2;
  vector[N3] treatment3;
  vector[N4] treatment4;
}
transformed data {
  vector[N1] inter1;
  vector[N2] inter2;
  vector[N3] inter3;
  vector[N4] inter4;
  inter1 <- treatment1 .* pre_test1;
  inter2 <- treatment2 .* pre_test2;
  inter3 <- treatment3 .* pre_test3;
  inter4 <- treatment4 .* pre_test4;
}
parameters {
  vector[4] beta1;
  vector[4] beta2;
  vector[4] beta3;
  vector[4] beta4;
  real<lower=0> sigma1;
  real<lower=0> sigma2;
  real<lower=0> sigma3;
  real<lower=0> sigma4;
} 
model {
  post_test1 ~ normal(beta1[1] + beta1[2] * treatment1 + beta1[3] * pre_test1 
                      + beta1[4] * inter1,sigma1);
  post_test2 ~ normal(beta2[1] + beta2[2] * treatment2 + beta2[3] * pre_test2
                      + beta2[4] * inter2,sigma2);
  post_test3 ~ normal(beta3[1] + beta3[2] * treatment3 + beta3[3] * pre_test3
                      + beta3[4] * inter3,sigma3);
  post_test4 ~ normal(beta4[1] + beta4[2] * treatment4 + beta4[3] * pre_test4
                      + beta4[4] * inter4,sigma4);
}
