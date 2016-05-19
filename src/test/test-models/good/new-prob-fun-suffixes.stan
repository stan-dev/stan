parameters {
  real y;
}
model {
  y ~ normal(0, 1);
  target += normal_lpdf(1 | -1.5, 2.9);
  target += poisson_lpmf(2 | 3);
  target += normal_lcdf(1 | 2, 3);
  target += normal_lccdf(1 | 2, 3);
  target += poisson_lcdf(1 | 2);
  target += poisson_lccdf(1 | 2);
}
