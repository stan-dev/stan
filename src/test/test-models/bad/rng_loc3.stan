functions {
  real foo_lpdf(real x) {
    return exponential_rng(x);
  }
}
model {
}
