functions {
  real unit_normal_rng() {
    return normal_rng(0,1);
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
generated quantities {
  real z;
  z <- unit_normal_rng();
}
