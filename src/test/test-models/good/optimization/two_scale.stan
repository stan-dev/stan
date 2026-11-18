/**
 * Two parameters with curvatures 1e12 and 1. When x1 starts close to its
 * optimum, its gradient still sets the length of the first step, which then
 * hardly moves x2. Optimum: (x1, x2) = (0, 5).
 */
parameters {
  real x1;
  real x2;
}
model {
  x1 ~ normal(0, 1e-6);
  x2 ~ normal(5, 1);
}
