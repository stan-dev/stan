functions {
  real foo_lpmf(int x) {
    return exponential_rng(x);
  }
}
model {
}
