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
              "Parse Error.  CDF already defined for poisson");
  test_throws("redefine-cdf2",
              "Parse Error.  CDF already defined for poisson");
  test_throws("redefine-cdf3",
              "Parse Error.  CDF already defined for foo");
}
TEST(langParser, ccdfRedefine) {
  test_throws("redefine-ccdf1",
              "Parse Error.  CCDF already defined for poisson");
  test_throws("redefine-ccdf2",
              "Parse Error.  CCDF already defined for poisson");
  test_throws("redefine-ccdf3",
              "Parse Error.  CCDF already defined for foo");
}
TEST(langParser, lpmfReal) {
  test_throws("real-pmf",
              "Parse Error.  Probability mass functions require"
              " integer variates (first argument). Found type = real");
}
TEST(langParser, lpdfReal) {
  test_throws("real-pdf",
              "Parse Error.  Probability density functions require"
              " real variates (first argument). Found type = int");
}
TEST(langParser, badArgs) {
  test_throws("prob-fun-args-no-bar",
              "Probabilty functions with suffixes _lpdf, _lpmf,"
                  " _lcdf, and _lccdf,",
              "require a vertical bar (|) between the first two"
                  " arguments.");
}
