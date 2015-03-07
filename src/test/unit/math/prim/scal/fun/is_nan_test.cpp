#include <stan/math/prim/scal/fun/is_nan.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, is_nan) {
  using stan::math::is_nan;
  double infinity = std::numeric_limits<double>::infinity();
  double nan = std::numeric_limits<double>::quiet_NaN();
  double min = std::numeric_limits<double>::min();
  double max = std::numeric_limits<double>::max();
  EXPECT_TRUE(stan::math::is_nan(nan));
  EXPECT_FALSE(stan::math::is_nan(infinity));
  EXPECT_FALSE(stan::math::is_nan(0));
  EXPECT_FALSE(stan::math::is_nan(1));
  EXPECT_FALSE(stan::math::is_nan(min));
  EXPECT_FALSE(stan::math::is_nan(max));
}

