#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, lbeta) {
  using stan::math::lbeta;
  
  EXPECT_FLOAT_EQ(0.0, lbeta(1.0,1.0));
  EXPECT_FLOAT_EQ(2.981361, lbeta(0.1,0.1));
  EXPECT_FLOAT_EQ(-4.094345, lbeta(3.0,4.0));
  EXPECT_FLOAT_EQ(-4.094345, lbeta(4.0,3.0));
}

TEST(MathFunctions, lbeta_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::lbeta(nan, 1.0));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::lbeta(1.0, nan));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::lbeta(nan, nan));
}
