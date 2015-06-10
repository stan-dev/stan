#include <stan/math/prim/scal/fun/logical_gt.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathFunctions,logical_gt) {
  using stan::math::logical_gt;
  EXPECT_TRUE(logical_gt(1,0));
  EXPECT_TRUE(logical_gt(2,1.00));
  EXPECT_TRUE(logical_gt(2.0,1));
  EXPECT_TRUE(logical_gt(0,-1));

  EXPECT_FALSE(logical_gt(1,1));
  EXPECT_FALSE(logical_gt(5.7,5.7));
  EXPECT_FALSE(logical_gt(-5.7,9.0));
  EXPECT_FALSE(logical_gt(0,0.0));
}

TEST(MathFunctions, logical_gt_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  EXPECT_FALSE(stan::math::logical_gt(1.0, nan));
  EXPECT_FALSE(stan::math::logical_gt(nan, 2.0));
  EXPECT_FALSE(stan::math::logical_gt(nan, nan));
}
