//Testing functions ending in _log, _lp and _rng
functions {
  void test_lp(real a) {
    a ~ normal(0, 1);
  }

  real test_rng(real a) {
    return normal_rng(a, 1);
  }

  real test_lpdf(real a, real b) {
    return normal_lpdf(a | b, 1);
  }  
}