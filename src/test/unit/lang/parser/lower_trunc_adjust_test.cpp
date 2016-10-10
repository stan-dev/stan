#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParser, lowerTruncAdjust) {
  test_parsable("lower-trunc-discrete");
  // built-in T[L, ]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-log_sum_exp(poisson_ccdf_log(L, lambda), poisson_log(L, lambda)));");
  // built-in T[L, U]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-log_sum_exp(log_diff_exp(poisson_cdf_log(U, lambda), poisson_cdf_log(L, lambda)), poisson_log(L, lambda)));");
  // built-in T[ , U]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-poisson_cdf_log(U, lambda));");
  // user-defined T[L, ]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-log_sum_exp(foo_lccdf(L, lambda, pstream__), foo_lpmf(L, lambda, pstream__)));");
  // user-defined T[L, U]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-log_sum_exp(log_diff_exp(foo_lcdf(U, lambda, pstream__), foo_lcdf(L, lambda, pstream__)), foo_lpmf(L, lambda, pstream__)));");
  // user-defined T[ , U]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-foo_lcdf(U, lambda, pstream__));");
}
