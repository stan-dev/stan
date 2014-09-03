#include <stan/math/functions/log_inv_logit.hpp>
#include <stan/math/functions/inv_logit.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, log_inv_logit) {
  using stan::math::log_inv_logit;
  using std::log;
  using stan::math::inv_logit;

  EXPECT_FLOAT_EQ(log(inv_logit(-7.2)), log_inv_logit(-7.2));
  EXPECT_FLOAT_EQ(log(inv_logit(0.0)), log_inv_logit(0.0));
  EXPECT_FLOAT_EQ(log(inv_logit(1.9)), log_inv_logit(1.9));
}

TEST(MathFunctions, log_inv_logit_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::log_inv_logit(nan));
}
