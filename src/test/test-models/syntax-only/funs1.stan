functions {
  void unit_normal_lp(real u) {
    increment_log_prob(normal_log(u,0,1));
    u ~ uniform(-100,100);
  }
}
parameters {
  real y;
}
model {
  unit_normal_lp(y);
}
