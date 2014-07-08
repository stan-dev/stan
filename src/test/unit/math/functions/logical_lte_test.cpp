#include "stan/math/functions/logical_lte.hpp"
#include <gtest/gtest.h>

TEST(MathFunctions,logical_lte) {
  using stan::math::logical_lte;
  EXPECT_TRUE(logical_lte(0,1));
  EXPECT_TRUE(logical_lte(1.0,2.0));
  EXPECT_TRUE(logical_lte(1, 2.0));
  EXPECT_TRUE(logical_lte(-1, 0));
  EXPECT_TRUE(logical_lte(1,1));
  EXPECT_TRUE(logical_lte(5.7,5.7));

  EXPECT_FALSE(logical_lte(5.7,-9.0));
  EXPECT_FALSE(logical_lte(-1,-2));
}
