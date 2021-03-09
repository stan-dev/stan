parameters {
  real y;
}
model {
  target += normal_lpdf(y| 0, 1) + normal_lcdf(2| 0, 1)
            + normal_lccdf(3| 0, 1);
}

