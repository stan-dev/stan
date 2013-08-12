data {
  int<lower=0> N;
  int<lower=0,upper=1> switched[N];
  vector[N] dist;
}
transformed data {
  vector[N] dist100;         // rescaling
  dist100 <- dist / 100.0;   
}
parameters {
  vector[2] beta;
}
model {
  switched ~ bernoulli_logit(beta[1] + beta[2] * dist100);
}
