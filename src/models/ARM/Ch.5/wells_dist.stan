data {
  int<lower=0> N;
  int<lower=0,upper=1> switched[N];
  vector[N] dist;
}
parameters {
  vector[2] beta;
}
model {
  switched ~ bernoulli_logit(beta[1] + beta[2] * dist);
}
