parameters {
  real<lower=0> x;
}

model {
  increment_log_prob(-sqrt(-x));
}