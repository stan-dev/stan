#include <stan/math/prim/scal/fun/log1m.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, log1m) {
  EXPECT_FLOAT_EQ(stan::math::log1p(-0.1),stan::math::log1m(0.1));
}

TEST(MathFunctions, log1m_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::log1m(nan));
}
