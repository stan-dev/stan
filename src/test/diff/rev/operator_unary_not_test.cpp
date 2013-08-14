#include <stan/diff/rev/operator_unary_not.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,not_a) {
  AVAR a(6.0);
  EXPECT_EQ(0, !a);
  AVAR b(0.0);
  EXPECT_EQ(1, !b);
}
