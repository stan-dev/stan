#include "stan/math/functions/inv_square.hpp"
#include <gtest/gtest.h>
#include <stan/math/constants.hpp>

TEST(MathsSpecialFunctions, inv_square) {
  double y = 2.0;
  EXPECT_FLOAT_EQ(1 / (y * y), stan::math::inv_square(y));

  y = 0.0;
  EXPECT_FLOAT_EQ(stan::math::positive_infinity(),stan::math::inv_square(y));

  y = -32.7;
  EXPECT_FLOAT_EQ(1 / (y * y), stan::math::inv_square(y));
}
