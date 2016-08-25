parameters {
  real y;
}
model {
  target += normal_lpdf(y, 0, 1);
}
