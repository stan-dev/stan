#include "stan/math/functions/inv_sqrt.hpp"
#include <gtest/gtest.h>

TEST(MathsSpecialFunctions, inv_square) {
  double y = 4.0;
  EXPECT_FLOAT_EQ(1 / 2.0, stan::math::inv_sqrt(y));

  y = 25.0;
  EXPECT_FLOAT_EQ(1 / 5.0, stan::math::inv_sqrt(y));
}
