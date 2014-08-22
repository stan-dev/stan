#include <stan/math/functions/fma.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, fma) {
  using stan::math::fma;
  
  EXPECT_FLOAT_EQ(5.0, fma(1.0,2.0,3.0));
  EXPECT_FLOAT_EQ(10.0, fma(2.0,3.0,4.0));
}

TEST(MathFunctions, fma_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::fma(1.0, 2.0, nan));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::fma(1.0, nan, 3.0));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::fma(1.0, nan, nan));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::fma(nan, 2.0, 3.0));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::fma(nan, 2.0, nan));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::fma(nan, nan, 3.0));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::fma(nan, nan, nan));
}
