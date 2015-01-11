#include <stan/math/functions/inv_cloglog.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, inv_cloglog) {
  EXPECT_EQ(1 - std::exp(-std::exp(3.7)), stan::math::inv_cloglog(3.7));
  EXPECT_EQ(1 - std::exp(-std::exp(0.0)), stan::math::inv_cloglog(0.0));
  EXPECT_EQ(1 - std::exp(-std::exp(-2.93)), stan::math::inv_cloglog(-2.93));
}

TEST(MathFunctions, inv_cloglog_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::inv_cloglog(nan));
}
