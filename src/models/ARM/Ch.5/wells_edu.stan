data {
  int<lower=0> N; 
  vector[N] arsenic;
  vector[N] dist;
  vector[N] educ;
  int<lower=0,upper=1> switc[N];
}
transformed data {
  vector[N] dist100;
  vector[N] educ4;

  dist100 <- dist / 100.0;
  educ4 <- educ / 4.0;
}
parameters {
  vector[4] beta;
} 
model {
  switc ~ bernoulli_logit(beta[1] + beta[2] * dist100 
                            + beta[3] * arsenic + beta[4] * educ4);
}
