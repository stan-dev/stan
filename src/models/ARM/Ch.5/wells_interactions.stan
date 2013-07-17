data {
  int<lower=0> N; 
  vector[N] dist;
  vector[N] arsenic;
  int<lower=0,upper=1> switc[N];
}
transformed data {
  vector[N] dist100;
  vector[N] inter;
  dist100 <- dist / 100.0;
  inter <- dist100 .* arsenic;
}
parameters {
  vector[4] beta;
} 
model {
  switc ~ bernoulli_logit(beta[1] + beta[2] * dist100
                            + beta[3] * arsenic + beta[4] * inter);
}
