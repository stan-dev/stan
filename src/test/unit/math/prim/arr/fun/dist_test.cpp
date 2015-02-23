#include <stan/math/prim/arr/fun/dist.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, dist) {
  std::vector<double> x(3), y(3);
  x[0] = 2.33;
  x[1] = 8.88;
  x[2] = 9.81;
  y[0] = 2.46;
  y[1] = 4.45;
  y[2] = 1.03;

  EXPECT_FLOAT_EQ(9.835151, stan::math::dist(x,y));
  EXPECT_FLOAT_EQ(9.835151, stan::math::dist(y,x));
  EXPECT_FLOAT_EQ(0.0, stan::math::dist(x,x));
}

TEST(MathFunctions, dist_nan) {
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
               stan::math::dist(x, y));
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::dist(y, x));


  x[0] = nan;
  x[1] = nan;
  x[2] = nan;
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::dist(x, y));
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::dist(y, x));


  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::dist(x, x));
}
