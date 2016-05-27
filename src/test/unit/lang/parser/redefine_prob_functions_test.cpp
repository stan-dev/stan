#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParser, probFunRedefine) {
  test_throws("redefine-prob1",
              "Parse Error.  Probability function already defined for poisson");
  test_throws("redefine-prob2",
              "Parse Error.  Function system defined, name=poisson_lpmf");
  test_throws("redefine-prob3",
              "Parse Error.  Probability function already defined for foo");
}
TEST(langParser, cdfRedefine) {
  test_throws("redefine-cdf1",
               "Parse Error.  Function system defined, name=poisson_cdf_log");
  test_throws("redefine-cdf2",
              "Parse Error.  Function system defined, name=poisson_lcdf");
  test_throws("redefine-cdf3",
              "Parse Error.  CDF already defined for foo");
}
TEST(langParser, ccdfRedefine) {
  test_throws("redefine-ccdf1",
               "Parse Error.  Function system defined, name=poisson_ccdf_log");
  test_throws("redefine-ccdf2",
              "Parse Error.  Function system defined, name=poisson_lccdf");
  test_throws("redefine-ccdf3",
              "Parse Error.  CCDF already defined for foo");
}
