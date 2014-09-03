#include <stan/math/functions/falling_factorial.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, falling_factorial) {
  using stan::math::falling_factorial;
  
  EXPECT_FLOAT_EQ(4, falling_factorial(4.0,3));
  EXPECT_FLOAT_EQ(0.25, falling_factorial(3.0,4));
  EXPECT_THROW(falling_factorial(-1, 4), std::domain_error);
}

TEST(MathFunctions, falling_factorial_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::falling_factorial(4.0, nan));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::falling_factorial(nan, 4.0));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::falling_factorial(nan, nan));
}
