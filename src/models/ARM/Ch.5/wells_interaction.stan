data {
  int<lower=0> N;
  int<lower=0,upper=1> switched[N];
  vector[N] dist;
  vector[N] arsenic;
}
transformed data {
  vector[N] dist100;         // rescaling
  vector[N] inter;           // interaction
  dist100 <- dist / 100.0;
  inter   <- dist100 .* arsenic;
}
parameters {
  vector[4] beta;
}
model {
  switched ~ bernoulli_logit(beta[1] + beta[2] * dist100 
                             + beta[3] * arsenic + beta[4] * inter);
}
