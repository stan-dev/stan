#include <stan/math/functions/owens_t.hpp>
#include <boost/math/special_functions/owens_t.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, owens_t) {
  double a = 1.0;
  double b = 2.0;
  EXPECT_FLOAT_EQ(stan::math::owens_t(a,b), boost::math::owens_t(a,b));
}

TEST(MathFunctions, owens_t_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::owens_t(1.0, nan));
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::owens_t(nan, 2.0));
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::owens_t(nan, nan));
}

