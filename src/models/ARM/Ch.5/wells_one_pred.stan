data {
  int<lower=0> N; 
  vector[N] dist;
  int<lower=0,upper=1> switc[N];
}
parameters {
  vector[2] beta;
} 
model {
  for (n in 1:N)
    switc[n] ~ bernoulli(inv_logit(beta[1] + beta[2] * dist[n]));
}
