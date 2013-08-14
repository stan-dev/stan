parameters {
  real<lower=-1,upper=1> y;
}
model {
  increment_log_prob(log1m(fabs(y)));
}
