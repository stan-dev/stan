#include <stan/math/functions/multiply_log.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, multiply_log) {
  double a = 2.0;
  double b = 3.0;
  EXPECT_FLOAT_EQ(a * log(b), stan::math::multiply_log(a,b));

  a = 0.0;
  b = 0.0;
  EXPECT_FLOAT_EQ(0.0, stan::math::multiply_log(a,b)) << 
    "when a and b are both 0, the result should be 0";

  a = 1.0;
  b = -1.0;
  EXPECT_TRUE(std::isnan(stan::math::multiply_log(a,b))) << 
    "log(b) with b < 0 should result in NaN";
}

TEST(MathFunctions, multiply_log_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::multiply_log(2.0, nan));
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::multiply_log(nan, 3.0));
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::multiply_log(nan, nan));
}
