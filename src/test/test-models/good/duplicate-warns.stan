model {
  real foo;
  foo <- 1;
  increment_log_prob(0);
  foo = get_lp();
  foo = multiply_log(1, 1);
  foo = binomial_coefficient_log(1, 1);
  // deprecated distribution functions versions
  foo = normal_log(0.5, 0, 1);
  foo = normal_cdf_log(0.5, 0, 1);
  foo = normal_ccdf_log(0.5, 0, 1);
}
