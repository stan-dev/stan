/**
 * The target is linear in x, so the gradient is nonzero while the
 * Hessian is identically zero. Used to check that the Newton optimizer
 * still moves along a direction with no curvature.
 */
parameters {
  real x;
}
model {
  target += x;
}
