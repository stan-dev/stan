functions {
  void unit_normal_lp(real u) {
    increment_log_prob(normal_log(u,0,1));
  }
}
parameters {
  real y;
}
model {
  increment_log_prob(unit_normal_lp(y));
}
