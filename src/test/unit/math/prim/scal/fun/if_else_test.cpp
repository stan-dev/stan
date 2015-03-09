#include <stan/math/prim/scal/fun/if_else.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions, if_else) {
  using stan::math::if_else;
  unsigned int c = 5;
  double x = 1.0;
  double y = -1.0;
  EXPECT_FLOAT_EQ(x, if_else(c,x,y));
  c = 0;
  EXPECT_FLOAT_EQ(y, if_else(c,x,y));

  bool d = true;
  int u = 1;
  int v = -1;
  EXPECT_EQ(1.0, stan::math::if_else(d,u,v));
  d = false;
  EXPECT_FLOAT_EQ(-1.0, stan::math::if_else(d,u,v));

  EXPECT_FLOAT_EQ(1.2,if_else(true,1.2,12));
  EXPECT_FLOAT_EQ(12.0,if_else(false,1.2,12));

  EXPECT_FLOAT_EQ(1.0,if_else(true,1,12.3));
  EXPECT_FLOAT_EQ(12.3,if_else(false,1,12.3));
}

TEST(MathFunctions, if_else_promote) {
  using stan::math::if_else;
  double x = 2.5;
  int y = -1;
  EXPECT_FLOAT_EQ(2.5, if_else(true,x,y));
  EXPECT_FLOAT_EQ(-1.0, if_else(false,x,y));
}

TEST(MathFunctions, if_else_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_FLOAT_EQ(1.2, stan::math::if_else(true, 1.2, nan));
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::if_else(false, 1.2, nan));


  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::if_else(true, nan, 2.4));
  EXPECT_FLOAT_EQ(2.4, stan::math::if_else(false, nan, 2.4));

  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::if_else(true, nan, nan));
  EXPECT_PRED1(boost::math::isnan<double>,
               stan::math::if_else(false, nan, nan));
}
