data {
  int<lower=0> N; 
  vector[N] dist;
  int<lower=0,upper=1> switc[N];
}
transformed data {
  vector[N] dist100;

  dist100 <- dist / 100.0;
}
parameters {
  vector[2] beta;
} 
model {
  switc ~ bernoulli_logit(beta[1] + beta[2] * dist100);
}