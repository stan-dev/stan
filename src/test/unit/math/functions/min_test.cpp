#include <stan/math/functions/min.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, min) {
  EXPECT_FLOAT_EQ(0.5, stan::math::min(0.5, 1.0));
  EXPECT_FLOAT_EQ(0.5, stan::math::min(1.0, 0.5));
  EXPECT_FLOAT_EQ(-1.0, stan::math::min(-0.5, -1.0));
  EXPECT_FLOAT_EQ(-1.0, stan::math::min(-1.0, -0.5));
}

TEST(MathFunctions, min_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_FLOAT_EQ(0.0, stan::math::min(nan, 0.0));
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::min(0.0, nan));
}
