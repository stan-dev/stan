/**
 * The target does not depend on x, so the gradient and Hessian
 * are identically zero along that direction. Used to check that
 * the Newton optimizer handles a flat direction without producing
 * non-finite parameter values.
 */
parameters {
  real x;
}
model {
  target += 0.5;
}
