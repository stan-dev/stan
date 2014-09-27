#include <stan/math/functions/log1p.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>


TEST(MathFunctions, log1p) {
  double x;

  x = 0;
  EXPECT_FLOAT_EQ(0.0, stan::math::log1p(x));
  x = 0.0000001;
  EXPECT_FLOAT_EQ(0.0000001, stan::math::log1p(x));
  x = 0.001;
  EXPECT_FLOAT_EQ(0.0009995003, stan::math::log1p(x));
  x = 0.1;
  EXPECT_FLOAT_EQ(0.09531018, stan::math::log1p(x));
  x = 1;
  EXPECT_FLOAT_EQ(0.6931472, stan::math::log1p(x));
  x = 10;
  EXPECT_FLOAT_EQ(2.397895, stan::math::log1p(x));

  x = -0.0000001;
  EXPECT_FLOAT_EQ(-0.0000001, stan::math::log1p(x));
  x = -0.001;
  EXPECT_FLOAT_EQ(-0.0010005, stan::math::log1p(x));
  x = -0.1;
  EXPECT_FLOAT_EQ(-0.1053605, stan::math::log1p(x));
  x = -0.999;
  EXPECT_FLOAT_EQ(-6.907755, stan::math::log1p(x));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), stan::math::log1p(-1.0));
}

TEST(MathFunctions, log1p_exception) {
  using boost::math::isnan;
  using stan::math::log1p;
  EXPECT_TRUE(isnan(log1p(-10.0)));
}

TEST(MathFunctions, log1p_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::log1p(nan));
}
