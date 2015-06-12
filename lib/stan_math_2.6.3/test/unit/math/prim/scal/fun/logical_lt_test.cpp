#include <stan/math/prim/scal/fun/logical_lt.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions,logical_lt) {
  using stan::math::logical_lt;
  EXPECT_TRUE(logical_lt(0,1));
  EXPECT_TRUE(logical_lt(1.0,2.0));
  EXPECT_TRUE(logical_lt(1, 2.0));
  EXPECT_TRUE(logical_lt(-1, 0));

  EXPECT_FALSE(logical_lt(1,1));
  EXPECT_FALSE(logical_lt(5.7,5.7));
  EXPECT_FALSE(logical_lt(5.7,-9.0));
  EXPECT_FALSE(logical_lt(0,0.0));
}

TEST(MathFunctions, logical_lt_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_FALSE(stan::math::logical_lt(1.0, nan));
  EXPECT_FALSE(stan::math::logical_lt(nan, 2.0));
  EXPECT_FALSE(stan::math::logical_lt(nan, nan));
}
