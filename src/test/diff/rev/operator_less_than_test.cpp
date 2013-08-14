#include <stan/diff/rev/operator_less_than.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,a_lt_b) {
  AVAR a = 5.0;
  AVAR b = 6.0;
  EXPECT_TRUE(a < b);
  EXPECT_FALSE(b < a);
  AVAR c = 6.0;
  EXPECT_FALSE(b < c);
  EXPECT_FALSE(c < b);
}

TEST(DiffRev,a_lt_y) {
  AVAR a = 5.0;
  double y = 6.0;
  EXPECT_TRUE(a < y);
  EXPECT_FALSE(y < a);
  AVAR b = 6.0;
  EXPECT_FALSE(b < y);
  EXPECT_FALSE(y < b);
}

TEST(DiffRev,x_lt_b) {
  double x = 5.0;
  AVAR b = 6.0;
  EXPECT_TRUE(x < b);
  EXPECT_FALSE(b < x);
  double y = 6.0;
  EXPECT_FALSE(b < y);
  EXPECT_FALSE(y < b);
}
