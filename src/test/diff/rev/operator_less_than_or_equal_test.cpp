#include <stan/diff/rev/operator_less_than_or_equal.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRev,a_lte_b) {
  AVAR a = 5.0;
  AVAR b = 6.0;
  EXPECT_TRUE(a <= b);
  EXPECT_FALSE(b <= a);
  AVAR c = 6.0;
  EXPECT_TRUE(b <= c);
  EXPECT_TRUE(c <= b);
}

TEST(DiffRev,a_lte_y) {
  AVAR a = 5.0;
  double y = 6.0;
  EXPECT_TRUE(a <= y);
  EXPECT_FALSE(y <= a);
  AVAR c = 5.0;
  EXPECT_TRUE(a <= c);
  EXPECT_TRUE(c <= a);
}


TEST(DiffRev,x_lte_b) {
  double x = 5.0;
  AVAR b = 6.0;
  EXPECT_TRUE(x <= b);
  EXPECT_FALSE(b <= x);
  double y = 6.0;
  EXPECT_TRUE(b <= y);
  EXPECT_TRUE(y <= b);
}
