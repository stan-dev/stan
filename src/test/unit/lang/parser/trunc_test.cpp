#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(parserTruncTest, poisson_log_log) {
  test_throws("prob-poisson_log-trunc",
              "lower truncation not defined",
              "to poisson_log");
}
