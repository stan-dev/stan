#include <stan/math/functions/log_falling_factorial.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, log_falling_factorial) {
  using stan::math::log_falling_factorial;
  
  EXPECT_FLOAT_EQ(std::log(4.0), log_falling_factorial(4.0,3));
  EXPECT_FLOAT_EQ(std::log(0.25), log_falling_factorial(3.0,4));
  EXPECT_THROW(log_falling_factorial(-1, 4), std::domain_error);
}

TEST(MathFunctions, log_falling_factorial_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::log_falling_factorial(nan, 3));
}
