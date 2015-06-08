#include <stan/math/prim/scal/fun/logical_neq.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathSpecialFunctions,logical_neq) {
  using stan::math::logical_neq;
  EXPECT_TRUE(logical_neq(0,1));
  EXPECT_TRUE(logical_neq(1.0,0));
  EXPECT_TRUE(logical_neq(1, 2));
  EXPECT_TRUE(logical_neq(2.0, -1.0));

  EXPECT_FALSE(logical_neq(1,1));
  EXPECT_FALSE(logical_neq(5.7,5.7));
  EXPECT_FALSE(logical_neq(0,0.0));
}

TEST(MathFunctions, logical_neq_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_TRUE(stan::math::logical_neq(1.0, nan));
  EXPECT_TRUE(stan::math::logical_neq(nan, 2.0));
  EXPECT_TRUE(stan::math::logical_neq(nan, nan));
}
