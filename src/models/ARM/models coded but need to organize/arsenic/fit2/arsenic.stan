data {
  int<lower=0> N; 
  vector[N] dist;
  int<lower=0,upper=1> switch_w[N];
}
transformed data {
  vector[N] dist100;
  dist100 <- dist / 100;
}
parameters {
  vector[2] beta;
} 
model {
  for (n in 1:N)
    switch_w[n] ~ bernoulli(inv_logit(beta[1] + beta[2] * dist100[n]));
}
