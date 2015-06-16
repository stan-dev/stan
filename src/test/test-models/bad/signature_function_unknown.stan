data {
  vector[4] x;
}
parameters {
  vector[4] theta;
}
model {
  increment_log_prob(foo_whatev_log(x, theta));
}
