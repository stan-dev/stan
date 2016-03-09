/**
 * this one's for issue #1768, where there was a duplicate fun decl
 * because of the <false> instantiation of propto
 */
functions {
  real n_log(real y);

  real n_log(real y) {
    return -0.5 * square(y);
  }
}
parameters {
  real mu;
}
model {
  mu ~ n();
  increment_log_prob(n_log(mu));  // check both instantiations
}


