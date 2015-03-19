#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, binomial_coefficient_log) {
  using stan::math::binomial_coefficient_log;
  EXPECT_FLOAT_EQ(1.0, exp(binomial_coefficient_log(2.0,2.0)));
  EXPECT_FLOAT_EQ(2.0, exp(binomial_coefficient_log(2.0,1.0)));
  EXPECT_FLOAT_EQ(3.0, exp(binomial_coefficient_log(3.0,1.0)));
  EXPECT_NEAR(3.0, exp(binomial_coefficient_log(3.0,2.0)),0.0001);
}

TEST(MathFunctions, binomial_coefficient_log_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::binomial_coefficient_log(2.0, nan));
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::binomial_coefficient_log(nan, 2.0));
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::binomial_coefficient_log(nan, nan));
}
