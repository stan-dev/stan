data {
  int<lower=0> N; 
  vector[N] dist;
  vector[N] arsenic;
  vector[N] educ;
  int<lower=0,upper=1> switc[N];
}
transformed data {
  vector[N] dist100;
  vector[N] educ4;
  vector[N] inter_dist_ars;
  dist100 <- dist / 100.0;
  educ4 <- educ / 4.0;
  inter_dist_ars <- dist100 .* arsenic;
}
parameters {
  vector[5] beta;
} 
model {
  switc ~ bernoulli_logit(beta[1] + beta[2] * dist100 
                            + beta[3] * arsenic + beta[4] * educ4
                            + beta[5] * inter_dist_ars);
}
