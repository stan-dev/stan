#include <stan/agrad/rev/operator_greater_than_or_equal.hpp>
#include <test/agrad/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,a_gte_b) {
  AVAR a = 5.0;
  AVAR b = 6.0;
  EXPECT_TRUE(b >= a);
  EXPECT_FALSE(a >= b);
  AVAR c = 6.0;
  EXPECT_TRUE(b >= c);
  EXPECT_TRUE(c >= b);
}

TEST(AgradRev,a_gte_y) {
  AVAR a = 6.0;
  double y = 5.0;
  EXPECT_TRUE(a >= y);
  EXPECT_FALSE(y >= a);
  AVAR c = 6.0;
  EXPECT_TRUE(a >= c);
  EXPECT_TRUE(c >= a);
}

TEST(AgradRev,x_gte_b) {
  double x = 6.0;
  AVAR b = 5.0;
  EXPECT_TRUE(x >= b);
  EXPECT_FALSE(b >= x);
  double y = 5.0;
  EXPECT_TRUE(b >= y);
  EXPECT_TRUE(y >= b);
}
