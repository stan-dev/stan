#include <stan/math/prim/scal/fun/trigamma.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, trigamma) {
  using stan::math::trigamma;
  EXPECT_FLOAT_EQ(102.9757436100834515253246058208,trigamma(-2.1));
  EXPECT_FLOAT_EQ(stan::math::positive_infinity(), trigamma(-2.0));
  EXPECT_FLOAT_EQ(1.0e12, trigamma(0.000001));
  EXPECT_FLOAT_EQ(0.2616741772864245283534239, trigamma(4.3));
  EXPECT_FLOAT_EQ(0.07404026866401033683840011, trigamma(14.0));
}

TEST(MathFunctions, trigamma_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::trigamma(nan));
}
