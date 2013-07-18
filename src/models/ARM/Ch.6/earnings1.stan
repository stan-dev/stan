data {
  int<lower=0> N; 
  int<lower=0,upper=1> earn_pos[N];
  vector[N] height;
  vector[N] male;
} 
parameters {
  vector[3] beta;
  real<lower=0> sigma;
} 
model {
  for (n in 1:N)
    earn_pos[n] ~ bernoulli(inv_logit(beta[1] + beta[2] * height[n] 
                            + beta[3] * male[n]));
}
