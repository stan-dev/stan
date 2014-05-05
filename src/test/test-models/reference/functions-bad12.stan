functions {
  void badlp(real x) {
    increment_log_prob(normal_log(x,0,1));
    return;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
