data {
  int<lower=0> N;
  vector[N] arsenic;
  vector[N] dist;
  int<lower=0,upper=1> switched[N];
}
transformed data {
  vector[N] c_arsenic;
  vector[N] c_dist100;
  vector[N] inter;

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
