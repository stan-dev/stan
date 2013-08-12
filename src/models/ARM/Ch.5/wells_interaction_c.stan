data {
  int<lower=0> N;
  int<lower=0,upper=1> switched[N];
  vector[N] dist;
  vector[N] arsenic;
}
transformed data {
  vector[N] c_dist100;       // centering
  vector[N] c_arsenic;
  vector[N] inter;           // interaction
  c_dist100 <- (dist - mean(dist)) / 100.0;
  c_arsenic <- arsenic - mean(arsenic);
  inter     <- c_dist100 .* c_arsenic;
}
parameters {
  vector[4] beta;
}
model {
  switched ~ bernoulli_logit(beta[1] + beta[2] * c_dist100 + beta[3] * c_arsenic
                              + beta[4] * inter);
}
