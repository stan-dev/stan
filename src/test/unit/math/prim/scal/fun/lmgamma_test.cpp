#include <stan/math/prim/scal/fun/lmgamma.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, lmgamma) {
  unsigned int k = 1;
  double x = 2.5;
  double result = k * (k - 1) * log(boost::math::constants::pi<double>()) / 4.0;
  result += lgamma(x); // j = 1
  EXPECT_FLOAT_EQ(result, stan::math::lmgamma(k,x));

  k = 2;
  x = 3.0;
  result = k * (k - 1) * log(boost::math::constants::pi<double>()) / 4.0;
  result += lgamma(x); // j = 1
  result += lgamma(x + (1.0 - 2.0)/2.0); // j = 2
  EXPECT_FLOAT_EQ(result, stan::math::lmgamma(k,x));
}

TEST(MathFunctions, lmgamma_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::lmgamma(2, nan));
}
