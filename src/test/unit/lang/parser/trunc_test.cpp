#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(parserTruncTest, poisson_log_log) {
  test_throws("prob-poisson_log-trunc-low",
              "lower truncation not defined",
              "arguments to poisson_log");
  test_throws("prob-poisson_log-trunc-high",
              "upper truncation not defined",
              "arguments to poisson_log");
  test_throws("prob-poisson_log-trunc-both",
              "lower truncation not defined",
              "arguments to poisson_log");
}
