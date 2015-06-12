#include <stan/math/prim/scal/fun/log_sum_exp.hpp>
#include <stan/math/prim/arr/fun/log_sum_exp.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

void test_log_sum_exp(double a, double b) {
  using std::log;
  using std::exp;
  using stan::math::log_sum_exp;
  EXPECT_FLOAT_EQ(log(exp(a) + exp(b)),
                  log_sum_exp(a,b));
}

void test_log_sum_exp(const std::vector<double>& as) {
  using std::log;
  using std::exp;
  using stan::math::log_sum_exp;
  double sum_exp = 0.0;
  for (size_t n = 0; n < as.size(); ++n)
    sum_exp += exp(as[n]);
  EXPECT_FLOAT_EQ(log(sum_exp),
                  log_sum_exp(as));
}

TEST(MathFunctions, log_sum_exp) {
  using stan::math::log_sum_exp;
  std::vector<double> as;
  test_log_sum_exp(as);
  as.push_back(0.0);
  test_log_sum_exp(as);
  as.push_back(1.0);
  test_log_sum_exp(as);
  as.push_back(-1.0);
  test_log_sum_exp(as);
  as.push_back(-10000.0);
  test_log_sum_exp(as);
  
  as.push_back(10000.0);
  EXPECT_FLOAT_EQ(10000.0, log_sum_exp(as));
}

TEST(MathFunctions, log_sum_exp_2) {
  using stan::math::log_sum_exp;
  test_log_sum_exp(1.0,2.0);
  test_log_sum_exp(1.0,1.0);
  test_log_sum_exp(3.0,2.0);
  test_log_sum_exp(-20.0,12);
  test_log_sum_exp(-20.0,12);

  // exp(10000.0) overflows
  EXPECT_FLOAT_EQ(10000.0,log_sum_exp(10000.0,0.0));
  EXPECT_FLOAT_EQ(0.0,log_sum_exp(-10000.0,0.0));
}

TEST(MathFunctions, log_sum_exp_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::log_sum_exp(1.0, nan));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::log_sum_exp(nan, 1.0));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::log_sum_exp(nan, nan));
}
