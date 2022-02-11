parameters {
  real x;
}
model {
  target += -0.5 * square(x);
}

