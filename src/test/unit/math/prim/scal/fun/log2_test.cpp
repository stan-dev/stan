#include <stan/math/prim/scal/fun/log2.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, log2_fun) {
  using stan::math::log2;

  EXPECT_FLOAT_EQ(std::log(2.0), log2());
}

TEST(MathFunctions, log2) {
  using stan::math::log2;

  EXPECT_FLOAT_EQ(1.0, log2(2.0));
  EXPECT_FLOAT_EQ(2.0, log2(4.0));
  EXPECT_FLOAT_EQ(3.0, log2(8.0));
}

TEST(MathFunctions, log2_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::log2(nan));
}
