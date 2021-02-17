functions {
  real n_lpdf(real y);
  real n_lpdf(real y) {
    return -0.5 * square(y);
  }
}
parameters {
  real mu;
}
model {
  mu ~ n();
  target += n_lpdf(mu| );
}

