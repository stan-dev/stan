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
  real<lower=0> sigma;
} 
model {
  for (n in 1:N)
    switc[n] ~ normal(inv_logit(beta[1] + beta[2] * dist100[n]),sigma);
}