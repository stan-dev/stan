#include <stan/math/functions/logical_gte.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions,logical_gte) {
  using stan::math::logical_gte;
  EXPECT_TRUE(logical_gte(1,0));
  EXPECT_TRUE(logical_gte(2.0,1.0));
  EXPECT_TRUE(logical_gte(2.0, 1));
  EXPECT_TRUE(logical_gte(0, -1));
  EXPECT_TRUE(logical_gte(1,1));
  EXPECT_TRUE(logical_gte(5.7,5.7));

  EXPECT_FALSE(logical_gte(-9.0, 5.7));
  EXPECT_FALSE(logical_gte(-2,-1));
}

TEST(MathFunctions, logical_gte_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_FALSE(stan::math::logical_gte(1.0, nan));
  EXPECT_FALSE(stan::math::logical_gte(nan, 2.0));
  EXPECT_FALSE(stan::math::logical_gte(nan, nan));
}
