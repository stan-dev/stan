transformed data {
  int N = 2;
}
generated quantities {
  real theta = beta_rng(1, 1);
  real eta = beta_rng(10, 10);
}
