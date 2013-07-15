data {
  int<lower=0> N; 
  vector[N] dist;
  vector[N] arsenic;
  int<lower=0,upper=1> switch_w[N];
}
transformed data {
  vector[N] dist100;
  vector[N] inter;
  dist100 <- dist / 100;
  inter <- dist100 .* arsenic;
}
parameters {
  vector[4] beta;
} 
model {
  for (n in 1:N)
    switch_w[n] ~ bernoulli(inv_logit(beta[1] + beta[2] * dist100[n] 
                            + beta[3] * arsenic[n] + beta[4] * inter[n]));
}
