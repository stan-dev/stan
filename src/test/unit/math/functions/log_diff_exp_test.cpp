#include <stan/math/functions/log_diff_exp.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

void test_log_diff_exp(double a, double b) {
  using std::log;
  using std::exp;
  using stan::math::log_diff_exp;
  EXPECT_FLOAT_EQ(log(exp(a) - exp(b)),
                  log_diff_exp(a,b));
}

TEST(MathFunctions, log_diff_exp) {
  using stan::math::log_diff_exp;
  test_log_diff_exp(3.0,2.0);
  test_log_diff_exp(4.0,1.0);
  test_log_diff_exp(3.0,2.0);
  test_log_diff_exp(0,-2.1);
  test_log_diff_exp(-20.0,-23);
  test_log_diff_exp(-21.2,-32.1);
  EXPECT_NO_THROW(log_diff_exp(-20.0,12));
  EXPECT_NO_THROW(log_diff_exp(-20.0,-12.1));
  EXPECT_NO_THROW(log_diff_exp(120.0,120.10));
  EXPECT_NO_THROW(log_diff_exp(-20.0,10.2));
  EXPECT_NO_THROW(log_diff_exp(10,11));
  EXPECT_NO_THROW(log_diff_exp(10,10));
  EXPECT_NO_THROW(log_diff_exp(-10.21,-10.21));

  // exp(10000.0) overflows
  EXPECT_FLOAT_EQ(10000.0,log_diff_exp(10000.0,0.0));
  EXPECT_FLOAT_EQ(0.0,log_diff_exp(0.0,-10000.0));
}

TEST(MathFunctions, log_diff_exp_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::log_diff_exp(3.0, nan));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::log_diff_exp(nan, 2.0));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::log_diff_exp(nan, nan));
}
