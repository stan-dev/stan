parameters {
  real x;
}

model {
  increment_log_prob(-0.5 * square(x));
}