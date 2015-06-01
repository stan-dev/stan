#include <stan/math/prim/scal/fun/log1m_exp.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, log1m_exp) {
  using stan::math::log1m_exp;

  // exp(10000.0) overflows
  EXPECT_FLOAT_EQ(0,log1m_exp(-1e10));
  EXPECT_FLOAT_EQ(0,log1m_exp(-1000));
  EXPECT_FLOAT_EQ(-3.720076e-44,log1m_exp(-100));
  EXPECT_FLOAT_EQ(-4.540096e-05,log1m_exp(-10));
  EXPECT_FLOAT_EQ(-0.4586751,log1m_exp(-1));
  EXPECT_FLOAT_EQ(-2.352168,log1m_exp(-0.1));
  EXPECT_FLOAT_EQ(-11.51293,log1m_exp(-1e-5));
  EXPECT_FLOAT_EQ(-23.02585,log1m_exp(-1e-10));
  EXPECT_FLOAT_EQ(-46.0517,log1m_exp(-1e-20));
  EXPECT_FLOAT_EQ(-92.1034,log1m_exp(-1e-40));
  EXPECT_NO_THROW(log1m_exp(0));
  EXPECT_NO_THROW(log1m_exp(1));
}

TEST(MathFunctions, log1m_exp_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::log1m_exp(nan));
}
