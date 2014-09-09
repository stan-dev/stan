#include <stan/math/functions/log_rising_factorial.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, log_rising_factorial) {
  using stan::math::log_rising_factorial;
  
  EXPECT_FLOAT_EQ(std::log(120.0), log_rising_factorial(4.0,3));
  EXPECT_FLOAT_EQ(std::log(360.0), log_rising_factorial(3.0,4));
  EXPECT_THROW(log_rising_factorial(-1, 4),std::domain_error);
}

TEST(MathFunctions, log_rising_factorial_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::log_rising_factorial(nan, 3));
}
