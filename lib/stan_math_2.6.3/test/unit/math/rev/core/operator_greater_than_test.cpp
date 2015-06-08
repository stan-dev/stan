#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,a_gt_b) {
  AVAR a = 5.0;
  AVAR b = 6.0;
  EXPECT_TRUE(b > a);
  EXPECT_FALSE(a > b);
  AVAR c = 6.0;
  EXPECT_FALSE(b > c);
  EXPECT_FALSE(c > b);
}

TEST(AgradRev,a_gt_y) {
  AVAR a = 6.0;
  double y = 5.0;
  EXPECT_TRUE(a > y);
  EXPECT_FALSE(y > a);
  AVAR c = 6.0;
  EXPECT_FALSE(a > c);
  EXPECT_FALSE(c > a);
}

TEST(AgradRev,x_gt_b) {
  double x = 6.0;
  AVAR b = 5.0;
  EXPECT_TRUE(x > b);
  EXPECT_FALSE(b > x);
  double y = 5.0;
  EXPECT_FALSE(b > y);
  EXPECT_FALSE(y > b);
}

TEST(AgradRev, logical_gt_nan) {
  stan::math::var nan = std::numeric_limits<double>::quiet_NaN();
  stan::math::var a = 1.0;
  stan::math::var b = 2.0;
  double nan_dbl = std::numeric_limits<double>::quiet_NaN();

  EXPECT_FALSE(1.0 > nan);
  EXPECT_FALSE(nan > 2.0);
  EXPECT_FALSE(nan > nan);
  EXPECT_FALSE(a > nan);
  EXPECT_FALSE(nan > b);
  EXPECT_FALSE(a > nan_dbl);
  EXPECT_FALSE(nan_dbl > b);
}
