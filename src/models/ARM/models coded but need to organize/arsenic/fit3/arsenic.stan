data {
  int<lower=0> N; 
  vector[N] dist;
  vector[N] arsenic;
  int<lower=0,upper=1> switch_w[N];
}
transformed data {
  vector[N] dist100;
  dist100 <- dist / 100;
}
parameters {
  vector[3] beta;
} 
model {
  for (n in 1:N)
    switch_w[n] ~ bernoulli(inv_logit(beta[1] + beta[2] * dist100[n] 
                            + beta[3] * arsenic[n]));
}
