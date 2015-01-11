#include <stan/math/functions/dot.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, dot) {
  std::vector<double> x(3), y(3);
  x[0] = 2.33;
  x[1] = 8.88;
  x[2] = 9.81;
  y[0] = 2.46;
  y[1] = 4.45;
  y[2] = 1.03;

  EXPECT_FLOAT_EQ(55.3521, stan::math::dot(x,y));
  EXPECT_FLOAT_EQ(55.3521, stan::math::dot(y,x));
  EXPECT_FLOAT_EQ(180.5194, stan::math::dot(x,x));
  EXPECT_FLOAT_EQ(26.915, stan::math::dot(y,y));
}

TEST(MathFunctions, dot_nan) {
  std::vector<double> x(3), y(3);
  x[0] = 2.33;
  x[1] = 8.88;
  x[2] = 9.81;
  y[0] = 2.46;
  y[1] = 4.45;
  y[2] = 1.03;

  double nan = std::numeric_limits<double>::quiet_NaN();
  x[2] = nan;

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::dot(x, y));
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::dot(y, x));


  x[0] = nan;
  x[1] = nan;
  x[2] = nan;
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::dot(x, y));
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::dot(y, x));


  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::dot(x, x));
}
