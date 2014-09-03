#include <stan/math/functions/inv_sqrt.hpp>
#include <stan/math/constants.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, inv_sqrt) {
  double y = 4.0;
  EXPECT_FLOAT_EQ(1 / 2.0, stan::math::inv_sqrt(y));

  y = 25.0;
  EXPECT_FLOAT_EQ(1 / 5.0, stan::math::inv_sqrt(y));

  y = 0.0;
  EXPECT_FLOAT_EQ(stan::math::positive_infinity(),stan::math::inv_sqrt(y));

  y = -50.0;
  std::isnan(stan::math::inv_sqrt(y));
}

TEST(MathFunctions, inv_sqrt_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::inv_sqrt(nan));
}
