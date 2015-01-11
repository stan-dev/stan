#include <stan/math/functions/max.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, max) {
  EXPECT_FLOAT_EQ(1.0, stan::math::max(0.5, 1.0));
  EXPECT_FLOAT_EQ(1.0, stan::math::max(1.0, 0.5));
  EXPECT_FLOAT_EQ(-0.5, stan::math::max(-0.5, -1.0));
  EXPECT_FLOAT_EQ(-0.5, stan::math::max(-1.0, -0.5));
}

TEST(MathFunctions, max_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_FLOAT_EQ(0.0, stan::math::max(nan, 0.0));
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::max(0.0, nan));
}
