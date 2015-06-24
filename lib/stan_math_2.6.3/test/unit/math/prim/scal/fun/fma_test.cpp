#include <cmath>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

// this is just testing the nan behavior of the built-in fma
// there is no longer a stan::math::fma, just the agrad versions
// instead, the top-level ::fma should be used by including <cmath>

TEST(MathFunctions, fma) {
  EXPECT_FLOAT_EQ(5.0, fma(1.0,2.0,3.0));
  EXPECT_FLOAT_EQ(10.0, fma(2.0,3.0,4.0));
}

TEST(MathFunctions, fma_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               fma(1.0, 2.0, nan));

  EXPECT_PRED1(boost::math::isnan<double>,
               fma(1.0, nan, 3.0));

  EXPECT_PRED1(boost::math::isnan<double>,
               fma(1.0, nan, nan));

  EXPECT_PRED1(boost::math::isnan<double>,
               fma(nan, 2.0, 3.0));

  EXPECT_PRED1(boost::math::isnan<double>,
               fma(nan, 2.0, nan));

  EXPECT_PRED1(boost::math::isnan<double>,
               fma(nan, nan, 3.0));

  EXPECT_PRED1(boost::math::isnan<double>,
               fma(nan, nan, nan));
}
