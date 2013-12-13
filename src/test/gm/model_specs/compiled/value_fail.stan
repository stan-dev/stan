parameters {
  real x;
}

model {
  increment_log_prob(1 / x);
}
