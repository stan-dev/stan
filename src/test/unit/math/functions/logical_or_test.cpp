#include "stan/math/functions/logical_or.hpp"
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
