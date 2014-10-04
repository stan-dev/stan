#include <stan/math/functions/is_inf.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, is_inf) {
  using stan::math::is_inf;
  double infinity = std::numeric_limits<double>::infinity();
  double nan = std::numeric_limits<double>::quiet_NaN();
  double min = std::numeric_limits<double>::min();
  double max = std::numeric_limits<double>::max();
  EXPECT_TRUE(stan::math::is_inf(infinity));
  EXPECT_FALSE(stan::math::is_inf(nan));
  EXPECT_FALSE(stan::math::is_inf(0));
  EXPECT_FALSE(stan::math::is_inf(1));
  EXPECT_FALSE(stan::math::is_inf(min));
  EXPECT_FALSE(stan::math::is_inf(max));
}

