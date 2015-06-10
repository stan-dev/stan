#include <stan/math/prim/arr/fun/dot_self.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, dot_self) {
  std::vector<double> x(3), y(3);
  x[0] = 2.33;
  x[1] = 8.88;
  x[2] = 9.81;
  y[0] = 2.46;
  y[1] = 4.45;
  y[2] = 1.03;

  EXPECT_FLOAT_EQ(180.5194, stan::math::dot_self(x));
  EXPECT_FLOAT_EQ(26.915, stan::math::dot_self(y));
}

TEST(MathFunctions, dot_self_nan) {
  std::vector<double> x(3);
  x[0] = 2.33;
  x[1] = 8.88;
  x[2] = 9.81;

  double nan = std::numeric_limits<double>::quiet_NaN();
  x[2] = nan;

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::dot_self(x));

  x[0] = nan;
  x[1] = nan;
  x[2] = nan;
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::dot_self(x));
}
