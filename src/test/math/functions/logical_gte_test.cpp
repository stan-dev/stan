#include "stan/math/functions/logical_gte.hpp"
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
