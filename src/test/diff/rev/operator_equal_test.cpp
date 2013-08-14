#include <stan/diff/rev/operator_equal.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,a_eq_b) {
  AVAR a = 2.0;
  AVAR b = 2.0;
  EXPECT_TRUE(a == b);
  EXPECT_TRUE(b == a);
  AVAR c = 3.0;
  EXPECT_FALSE(a == c);
  EXPECT_FALSE(c == a);
}

TEST(DiffRev,x_eq_b) {
  double x = 2.0;
  AVAR b = 2.0;
  EXPECT_TRUE(x == b);
  EXPECT_TRUE(b == x);
  AVAR c = 3.0;
  EXPECT_FALSE(x == c);
  EXPECT_FALSE(c == x);
}

TEST(DiffRev,a_eq_y) {
  AVAR a = 2.0;
  double y = 2.0;
  EXPECT_TRUE(a == y);
  EXPECT_TRUE(y == a);
  double z = 3.0;
  EXPECT_FALSE(a == z);
  EXPECT_FALSE(z == a);
}
