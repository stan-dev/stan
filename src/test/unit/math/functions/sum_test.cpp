#include <stan/math/functions/sum.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, sum) {
  std::vector<double> x(3);
  
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = 3.0;
  
  EXPECT_FLOAT_EQ(6.0, stan::math::sum(x));
}

TEST(MathFunctions, sub_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> x(3);
  
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = nan;

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::sum(x));
}
