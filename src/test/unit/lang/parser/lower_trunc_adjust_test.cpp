#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

// these tests make sure the built-in and user-defined
// discrete and continuous distributions generate the right code
// if the exact form of the code changes, these tests must be changed

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
               "lp_accum__.add(-log_sum_exp(foo_lccdf(L, lambda, pstream__),"
               " foo_lpmf(L, lambda, pstream__)));");
  // user-defined T[L, U]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-log_sum_exp(log_diff_exp(foo_lcdf(U, lambda,"
               " pstream__), foo_lcdf(L, lambda, pstream__)), foo_lpmf(L,"
               " lambda, pstream__)));");
  // user-defined T[ , U]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-foo_lcdf(U, lambda, pstream__));");

  // user-defined (deprecated) T[L, ]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-log_sum_exp(bar_ccdf_log(L, lambda,"
               " pstream__), bar_log(L, lambda, pstream__)));");
  // user-defined (deprecated) T[L, U]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-log_sum_exp(log_diff_exp(bar_cdf_log(U,"
               " lambda, pstream__), bar_cdf_log(L, lambda, pstream__)),"
               " bar_log(L, lambda, pstream__)));");
  // user-defined (deprecated) T[ , U]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-bar_cdf_log(U, lambda, pstream__));");

  // built-in continuous T[L, ]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-normal_ccdf_log(L, 0, 1));");

  // built-in continuous T[ , U]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-normal_cdf_log(U, 0, 1));");

  // built-in continuous T[L, U]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-log_diff_exp(normal_cdf_log(U, 0, 1),"
               " normal_cdf_log(L, 0, 1)));");

  // user-defined continuous T[L, ]
  expect_match("lower-trunc-discrete",
               " lp_accum__.add(-baz_lccdf(L, lambda, pstream__));");

  // user-defined continuous T[ , U]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-log_diff_exp(baz_lcdf(U, lambda,"
               " pstream__), baz_lcdf(L, lambda, pstream__)));");

  // user-defined continuous T[L, U]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-baz_lcdf(U, lambda, pstream__));");

  // user-defined, deprecated continuous T[L, ]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-quux_ccdf_log(L, lambda,"
               " pstream__));");

  // user-defined, deprecated continuous T[ , U]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-log_diff_exp(quux_cdf_log(U,"
               " lambda, pstream__), quux_cdf_log(L, lambda,"
               " pstream__)));");

  // user-defined, deprecated continuous T[L, U]
  expect_match("lower-trunc-discrete",
               "lp_accum__.add(-quux_cdf_log(U, lambda,"
               " pstream__));");


}
