#include <stan/math/functions/log1p_exp.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, log1p_exp) {
  using stan::math::log1p_exp;

  // exp(10000.0) overflows
  EXPECT_FLOAT_EQ(10000.0,log1p_exp(10000.0));
  EXPECT_FLOAT_EQ(0.0,log1p_exp(-10000.0));
}

TEST(MathFunctions, log1p_exp_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::log1p_exp(nan));
}
