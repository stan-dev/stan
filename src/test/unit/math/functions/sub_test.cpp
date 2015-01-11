#include <stan/math/functions/sub.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, sub) {
  std::vector<double> x(3), y(3), result(3);
  
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = 3.0;
  
  y[0] = 0.5;
  y[1] = 1.0;
  y[2] = 4.0;
  
  EXPECT_NO_THROW(stan::math::sub(x, y, result));
  EXPECT_FLOAT_EQ(0.5, result[0]);
  EXPECT_FLOAT_EQ(1.0, result[1]);
  EXPECT_FLOAT_EQ(-1.0, result[2]);
}

TEST(MathFunctions, sub_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> x(3), y(3), result(3);
  
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = nan;
  
  y[0] = 0.5;
  y[1] = 1.0;
  y[2] = 4.0;
  
  EXPECT_NO_THROW(stan::math::sub(x, y, result));
  EXPECT_FLOAT_EQ(0.5, result[0]);
  EXPECT_FLOAT_EQ(1.0, result[1]);
  EXPECT_PRED1(boost::math::isnan<double>,
               result[2]);


  EXPECT_NO_THROW(stan::math::sub(y, x, result));
  EXPECT_FLOAT_EQ(-0.5, result[0]);
  EXPECT_FLOAT_EQ(-1.0, result[1]);
  EXPECT_PRED1(boost::math::isnan<double>,
               result[2]);

  
  y[2] = nan;
  EXPECT_NO_THROW(stan::math::sub(x, y, result));
  EXPECT_FLOAT_EQ(0.5, result[0]);
  EXPECT_FLOAT_EQ(1.0, result[1]);
  EXPECT_PRED1(boost::math::isnan<double>,
               result[2]);

}
