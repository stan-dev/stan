functions {
  void unit_normal_lp(real u) {
    target += normal_lpdf(u| 0, 1);
    u ~ uniform(-100, 100);
  }
}
parameters {
  real y;
}
model {
  unit_normal_lp(y);
}

