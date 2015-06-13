#include <stan/math/prim/scal/fun/logical_or.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions,logical_or) {
  using stan::math::logical_or;
  EXPECT_TRUE(logical_or(1,0));
  EXPECT_TRUE(logical_or(0,1));
  EXPECT_TRUE(logical_or(1,1));
  EXPECT_TRUE(logical_or(5.7,-1.9));
  EXPECT_TRUE(logical_or(5.7,-1));

  EXPECT_FALSE(logical_or(0,0));
  EXPECT_FALSE(logical_or(0.0, 0.0));
  EXPECT_FALSE(logical_or(0.0, 0));
  EXPECT_FALSE(logical_or(0, 0.0));
}

TEST(MathFunctions, logical_or_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_TRUE(stan::math::logical_or(1.0, nan));
  EXPECT_TRUE(stan::math::logical_or(nan, 2.0));
  EXPECT_TRUE(stan::math::logical_or(nan, nan));
}
