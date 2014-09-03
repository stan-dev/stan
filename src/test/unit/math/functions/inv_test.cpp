#include <stan/math/functions/inv.hpp>
#include <stan/math/constants.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathsSpecialFunctions, inv) {
  double y = 2.0;
  EXPECT_FLOAT_EQ(1 / y, stan::math::inv(y));

  y = 0.0;
  EXPECT_FLOAT_EQ(stan::math::positive_infinity(),stan::math::inv(y));

  y = -32.7;
  EXPECT_FLOAT_EQ(1 / y, stan::math::inv(y));
}

TEST(MathFunctions, inv_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::inv(nan));
}
