#include "stan/math/functions/inv_square.hpp"
#include <gtest/gtest.h>

TEST(MathsSpecialFunctions, inv_square) {
  double y = 2.0;
  EXPECT_FLOAT_EQ(1 / (y * y), stan::math::inv_square(y));

  y = 0.0;
  std::isnan(stan::math::inv_square(y));

  y = -32.7;
  EXPECT_FLOAT_EQ(1 / (y * y), stan::math::inv_square(y));
}
