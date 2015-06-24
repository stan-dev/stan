data {
  vector[4] x;
}
parameters {
  vector[4] theta;
}
model {
  increment_log_prob(bernoulli_logit_log(x, theta));
}
