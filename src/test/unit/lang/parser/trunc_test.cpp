#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(parserTruncTest, poisson_log_log) {
  test_throws("prob-poisson_log-trunc-low",
              "Lower truncation not defined",
              "arguments to poisson_log");
  test_throws("prob-poisson_log-trunc-high",
              "Upper truncation not defined",
              "arguments to poisson_log");
  test_throws("prob-poisson_log-trunc-both",
              "Lower truncation not defined",
              "arguments to poisson_log");
}
