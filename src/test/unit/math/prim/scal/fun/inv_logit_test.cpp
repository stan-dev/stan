#include <stan/math/prim/scal/fun/inv_logit.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, inv_logit) {
  using stan::math::inv_logit;
  EXPECT_FLOAT_EQ(0.5, inv_logit(0.0));
  EXPECT_FLOAT_EQ(1.0/(1.0 + exp(-5.0)), inv_logit(5.0));
}

TEST(MathFunctions, inv_logit_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::inv_logit(nan));
}
