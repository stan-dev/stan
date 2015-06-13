#include <stan/math/prim/scal/fun/fdim.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, fdim_double) {
  using stan::math::fdim;

  EXPECT_FLOAT_EQ(1.0, fdim(3.0,2.0));
  EXPECT_FLOAT_EQ(0.0, fdim(2.0,3.0));

  EXPECT_FLOAT_EQ(2.5, fdim(4.5,2.0));
}

TEST(MathFunctions, fdim_int) {
  using stan::math::fdim;

  // promotes results to double
  EXPECT_FLOAT_EQ(1.0, fdim(int(3),int(2)));
  EXPECT_FLOAT_EQ(0.0, fdim(int(2),int(3)));
}

TEST(MathFunctions, fdim_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::fdim(3.0, nan));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::fdim(nan, 3.0));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::fdim(nan, nan));
}
